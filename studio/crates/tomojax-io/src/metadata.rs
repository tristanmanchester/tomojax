//! Interpret the public inspection report and the storage layout of committed NXtomo.
//!
//! Inspection intentionally omits some calibration. Never promote guessed voxel
//! sizes or uncorrected file pixels into a registered reconstruction or scan.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Value, json};

use studio_core::protocol::{ArrayDescriptor, Axis, Coordinates, Storage};

/// Suggested input kind, registered file arrays and the metadata-only report.
pub type HeaderMetadata = (String, BTreeMap<String, ArrayDescriptor>, Value);

/// Where an NXtomo entry keeps its frames and rotation angles, as the loader looks.
const STACKS: [&str; 4] = [
    "instrument/detector/data",
    "data/data",
    "data/projections",
    "projections",
];
const ANGLES: [&str; 3] = [
    "sample/rotation_angle",
    "sample/transformations/rotation_angle",
    "data/rotation_angle",
];

/// Fast first-image metadata for standard NXtomo layouts. No pixel datasets or
/// coordinate vectors are read; the full worker inspection remains an explicit audit.
pub fn quick_metadata(path: &Path) -> Result<Option<HeaderMetadata>, String> {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "h5" | "hdf5" | "hdf" | "nxs" | "nx") {
        return Ok(None);
    }
    let file =
        hdf5::File::open(path).map_err(|error| format!("Cannot read HDF5 headers: {error}"))?;
    if let Some(metadata) = native_metadata(&file, path)? {
        return Ok(Some(metadata));
    }
    let root = nxtomo_entry(&file)?.unwrap_or_else(|| "/entry".into());
    let at = |name: &str| format!("{root}/{name}");
    let first = |names: &[&str]| names.iter().map(|n| at(n)).find(|n| file.link_exists(n));
    if !file.link_exists(&root) {
        return Ok(None);
    }
    let projection_path = first(&STACKS);
    if projection_path.is_none() {
        return Ok(None);
    }
    let geometry_type = match file.group(&at("geometry")) {
        Ok(group) => string_attribute(&group, "type")?,
        Err(_) => None,
    };
    let mut projection = json!({"found":false,"path":null,"shape":null,"dtype":null});
    if let Some(dataset) = projection_path {
        let data = file.dataset(&dataset).map_err(|error| error.to_string())?;
        let stored = data.shape();
        let shape = match stored.as_slice() {
            [views, u] => vec![*views, 1, *u],
            [views, v, u] => vec![*views, *v, *u],
            _ => return Ok(None),
        };
        if shape.contains(&0) {
            return Err("The projection dataset has an empty dimension".into());
        }
        let dtype = data.dtype().map_err(|error| error.to_string())?;
        let bytes = shape
            .iter()
            .try_fold(dtype.size() as u64, |bytes, size| {
                bytes.checked_mul(*size as u64)
            })
            .ok_or("Projection byte count overflow")?;
        projection = json!({"found":true,"path":dataset,"shape":shape,"dtype":dtype.to_descriptor().map_err(|error|error.to_string())?.to_string(),
            "n_views":shape[0],"detector_shape":{"nv":shape[1],"nu":shape[2]},"storage_bytes":bytes});
    }
    let angle_path = first(&ANGLES);
    let angles = if let Some(dataset) = angle_path {
        let values = file.dataset(&dataset).map_err(|error| error.to_string())?;
        let shape = values.shape();
        if shape.len() != 1 {
            return Ok(None);
        }
        json!({"found":true,"path":dataset,"count":shape[0],"units":string_attribute(&values,"units")?})
    } else {
        json!({"found":false,"path":null,"count":null,"units":null})
    };
    let volume = json!({"found":false});
    // NXtomo image keys mark sample (0), flat (1) and dark (2) frames, and an
    // `image_key_control` beside them marks return and alignment images (-1) to
    // leave out. They are found as the worker finds them: at the detector, else
    // the file's only per-frame dataset named `image_key`.
    let frames = projection["n_views"].as_u64().map(|n| n as usize);
    let image_key = Some(at("instrument/detector/image_key"))
        .filter(|name| file.link_exists(name))
        .or_else(|| frames.and_then(|frames| only_per_frame(&file, "image_key", frames)));
    let key_found = image_key.is_some();
    let read_keys = |name: &str| -> Result<Option<Vec<i64>>, String> {
        if !file.link_exists(name) {
            return Ok(None);
        }
        file.dataset(name)
            .and_then(|keys| keys.read_raw::<i64>())
            .map(Some)
            .map_err(|error| error.to_string())
    };
    let (sample_count, first_sample) = if let Some(path) = &image_key {
        let folder = path.rsplit_once('/').map_or("", |(folder, _)| folder);
        let control = read_keys(&format!("{folder}/image_key_control"))?;
        let keys = file
            .dataset(path)
            .and_then(|d| d.read_raw::<i64>())
            .map_err(|e| e.to_string())?;
        let samples = keys.iter().enumerate().filter_map(|(i, key)| {
            (*key == 0
                && control
                    .as_ref()
                    .is_none_or(|values| values.get(i) != Some(&-1)))
            .then_some(i)
        });
        let mut samples = samples.peekable();
        let first = samples.peek().copied();
        (Some(samples.count()), first)
    } else {
        (None, None)
    };
    let flat_found = file.link_exists(&at("instrument/detector/data_white"));
    let dark_found = file.link_exists(&at("instrument/detector/data_dark"));
    let report = json!({"source":"header_only","input_path":path,
        "projection":projection,"angles":angles,"volume":volume,
        "geometry":{"found":geometry_type.is_some(),"type":geometry_type,"meta_found":false,"meta_keys":[]},
        "detector_metadata":{"found":false},
        "flats_darks":{"image_key_found":key_found,"image_key_path":image_key,
            "sample_count":sample_count,"first_sample":first_sample,
            "flats_present":if flat_found{Some(true)}else{None},"darks_present":if dark_found{Some(true)}else{None}},
        "alignment":{"found":false,"params_found":false}});
    let mut arrays = projection_arrays(&report, path)?;
    for array in arrays.values_mut() {
        // Raw HDF5 may interleave sample/calibration frames whose angle entries
        // are undefined. Until the loader selects samples, show stored indices.
        if key_found {
            array.axes[0].unit = "index".into();
            array.axes[0].coordinates = Coordinates::Uniform {
                origin: 0.,
                spacing: 1.,
            };
        }
    }
    Ok(Some(("frames".into(), arrays, report)))
}

/// Whether an HDF5 file reads any of its data from other files: through external
/// links, virtual datasets or external raw storage, as NXtomo masters often do.
pub fn reads_other_files(path: &Path) -> Result<bool, String> {
    let file = hdf5::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut groups = vec![(*file).clone()];
    let mut visited = 0usize;
    while let Some(group) = groups.pop() {
        // Hard-linked groups can form cycles; real files have far fewer groups.
        visited += 1;
        if visited > 100_000 {
            break;
        }
        let mut links = Vec::new();
        group
            .iter_visit_default(|name, info| {
                links.push((name.to_owned(), info.link_type));
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        for (name, kind) in links {
            match kind {
                hdf5::LinkType::External => return Ok(true),
                hdf5::LinkType::Hard => {
                    if let Ok(child) = group.group(&name) {
                        groups.push(child);
                    } else if let Ok(dataset) = group.dataset(&name) {
                        let create = dataset.dcpl().map_err(|e| e.to_string())?;
                        if create.layout() == hdf5::dataset::Layout::Virtual
                            || !create.external().is_empty()
                        {
                            return Ok(true);
                        }
                    }
                }
                hdf5::LinkType::Soft => {}
            }
        }
    }
    Ok(false)
}

/// The path of the file's only dataset named `name` holding one value per frame.
fn only_per_frame(file: &hdf5::File, name: &str, frames: usize) -> Option<String> {
    let mut found = Vec::new();
    let mut groups = vec![file.group("/").ok()?];
    while let Some(group) = groups.pop() {
        let mut links = Vec::new();
        let _ = group.iter_visit_default(|link, info| {
            if info.link_type == hdf5::LinkType::Hard {
                links.push(link.to_owned());
            }
            Ok(())
        });
        for link in links {
            if let Ok(child) = group.group(&link) {
                groups.push(child);
            } else if link == name
                && let Ok(dataset) = group.dataset(&link)
                && dataset.shape() == [frames]
            {
                found.push(dataset.name());
            }
        }
    }
    (found.len() == 1).then(|| found.remove(0))
}

pub fn suggested_kind(report: &Value) -> &'static str {
    if let Some(kind) = report.pointer("/record/kind").and_then(Value::as_str) {
        match kind {
            "scan"
                if report
                    .pointer("/record/arrays/projections/domain")
                    .and_then(Value::as_str)
                    == Some("counts") =>
            {
                return "frames";
            }
            "scan" => return "scan",
            "reconstruction" => return "reconstruction",
            "alignment" => return "alignment",
            "simulation" => return "simulation",
            "feature" => return "feature",
            _ => {}
        }
    }
    let keys = report
        .pointer("/geometry/meta_keys")
        .and_then(Value::as_array);
    let has = |key: &str| keys.is_some_and(|items| items.iter().any(|v| v.as_str() == Some(key)));
    if report
        .pointer("/volume/method")
        .is_some_and(Value::is_string)
    {
        "reconstruction"
    } else if report.pointer("/volume/found").and_then(Value::as_bool) == Some(true)
        && has("volume_role")
        && has("simulation_info")
    {
        "simulation"
    } else if has("alignment_info") {
        // An alignment is its record; a scan saved with a pose table is still a
        // scan (TomoJAX loads it as one and has no alignment record to load).
        "alignment"
    } else if report.pointer("/volume/found").and_then(Value::as_bool) == Some(true) {
        "reconstruction"
    } else if has("corrections") {
        // Corrected scans retain their original acquisition facts when saved.
        // Those facts do not turn the persisted line integrals back into counts.
        "scan"
    } else if report
        .pointer("/flats_darks/flats_present")
        .and_then(Value::as_bool)
        == Some(true)
        || report
            .pointer("/flats_darks/darks_present")
            .and_then(Value::as_bool)
            == Some(true)
        || has("acquisition")
    {
        "frames"
    } else {
        "scan"
    }
}

/// Read only small native HDF5 metadata, keeping existing volumes in place.
pub fn inspection_arrays(
    report: &Value,
    path: &Path,
) -> Result<BTreeMap<String, ArrayDescriptor>, String> {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "h5" | "hdf5" | "hdf" | "nxs" | "nx") {
        return Ok(BTreeMap::new());
    }
    let file =
        hdf5::File::open(path).map_err(|error| format!("Cannot read dataset metadata: {error}"))?;
    if let Some((_, arrays, _)) = native_metadata(&file, path)? {
        return Ok(arrays);
    }
    projection_arrays(report, path)
}

/// Direct file measurements, prior to any worker loader corrections or view selection.
fn projection_arrays(
    report: &Value,
    path: &Path,
) -> Result<BTreeMap<String, ArrayDescriptor>, String> {
    let mut arrays = BTreeMap::new();
    if report.pointer("/projection/found").and_then(Value::as_bool) != Some(true) {
        return Ok(arrays);
    }
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "h5" | "hdf5" | "hdf" | "nxs" | "nx") {
        return Ok(arrays);
    }
    let projection = &report["projection"];
    let dataset = projection["path"]
        .as_str()
        .filter(|p| p.starts_with('/'))
        .ok_or("Inspection did not identify an HDF5 projection dataset")?;
    let dimensions = projection["shape"]
        .as_array()
        .ok_or("Inspection has no projection shape")?;
    let shape: Vec<usize> = dimensions
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("Invalid projection dimension")
        })
        .collect::<Result<_, _>>()?;
    if shape.len() != 3 || shape.contains(&0) {
        return Err("Projection shape must be nonempty (view, v, u)".into());
    }
    // Keep mixed raw detector frames (including calibrations) distinct from sample views.
    let interleaved = report
        .pointer("/flats_darks/sample_count")
        .and_then(Value::as_u64)
        .is_some_and(|samples| samples != shape[0] as u64);
    let stored = hdf5::File::open(path).map_err(|error| error.to_string())?;
    let squeezed = |dims: &[usize]| dims.iter().copied().filter(|&n| n != 1).collect::<Vec<_>>();
    let stored_shape = stored.dataset(dataset).map(|d| d.shape()).ok();
    if stored_shape.as_deref().map(squeezed) != Some(squeezed(&shape)) {
        return Ok(arrays);
    }
    let file = path.to_string_lossy().into_owned();
    let angles = &report["angles"];
    // The unit the stored angles carry, not the report's converted one.
    let stored_unit = angles["path"]
        .as_str()
        .and_then(|dataset| stored.dataset(dataset).ok())
        .and_then(|dataset| string_attribute(&dataset, "units").ok().flatten())
        .map(|unit| match unit.as_str() {
            "deg" | "degrees" => "degree".to_owned(),
            "rad" | "radians" => "radian".to_owned(),
            _ => unit,
        });
    let view = match (angles["path"].as_str(), stored_unit) {
        (Some(dataset), Some(unit))
            if !interleaved
                && dataset.starts_with('/')
                && angles["count"].as_u64() == Some(shape[0] as u64) =>
        {
            Axis {
                name: "view".into(),
                frame: "acquisition".into(),
                unit,
                coordinates: Coordinates::Coordinates {
                    values: Storage::Hdf5 {
                        path: file.clone(),
                        dataset: dataset.into(),
                    },
                },
            }
        }
        _ => Axis {
            name: if interleaved { "frame" } else { "view" }.into(),
            frame: "acquisition".into(),
            unit: "index".into(),
            coordinates: Coordinates::Uniform {
                origin: 0.0,
                spacing: 1.0,
            },
        },
    };
    let detector = &report["detector_metadata"];
    let center = detector["center"].as_array();
    let mut axes = vec![view];
    for (name, index, pitch) in [("v", 1, "dv"), ("u", 0, "du")] {
        let dimension = if name == "v" { shape[1] } else { shape[2] };
        let calibrated = detector[pitch]
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.0)
            .zip(
                center
                    .and_then(|c| c.get(index))
                    .and_then(Value::as_f64)
                    .filter(|v| v.is_finite()),
            );
        let (unit, origin, spacing) = match calibrated {
            Some((spacing, center)) => (
                "scan_length",
                center - (dimension - 1) as f64 * spacing / 2.0,
                spacing,
            ),
            None => ("pixel", 0.0, 1.0),
        };
        axes.push(Axis {
            name: name.into(),
            frame: "detector".into(),
            unit: unit.into(),
            coordinates: Coordinates::Uniform { origin, spacing },
        });
    }
    arrays.insert(
        if interleaved { "counts" } else { "projections" }.into(),
        ArrayDescriptor {
            storage: Storage::Hdf5 {
                path: file,
                dataset: dataset.into(),
            },
            shape,
            dtype: projection["dtype"]
                .as_str()
                .ok_or("Inspection has no stored dtype")?
                .into(),
            axes,
            unit: "unspecified".into(),
            domain: if interleaved {
                "counts"
            } else {
                "stored_measurements"
            }
            .into(),
            fixed: Vec::new(),
            columns: Vec::new(),
        },
    );
    Ok(arrays)
}

fn string_attribute(group: &hdf5::Location, name: &str) -> Result<Option<String>, String> {
    if !group
        .attr_names()
        .map_err(|error| error.to_string())?
        .iter()
        .any(|key| key == name)
    {
        return Ok(None);
    }
    let attribute = group.attr(name).map_err(|error| error.to_string())?;
    text(&attribute, name).map(Some)
}

/// A scalar string, as NeXus writers store them: variable- or fixed-length.
fn text(value: &hdf5::Container, name: &str) -> Result<String, String> {
    use hdf5::types::{FixedAscii, FixedUnicode, VarLenAscii, VarLenUnicode};
    let text = if let Ok(value) = value.read_scalar::<VarLenUnicode>() {
        value.as_str().to_owned()
    } else if let Ok(value) = value.read_scalar::<VarLenAscii>() {
        value.as_str().to_owned()
    } else if let Ok(value) = value.read_scalar::<FixedUnicode<4096>>() {
        value.as_str().to_owned()
    } else {
        value
            .read_scalar::<FixedAscii<4096>>()
            .map_err(|error| format!("Cannot decode {name}: {error}"))?
            .as_str()
            .to_owned()
    };
    if text.len() > 1024 * 1024 {
        return Err(format!("The {name} metadata exceed 1 MiB"));
    }
    Ok(text)
}

/// The file's NXtomo entry, found as the loader finds it (GDA writes /entry/tomo_entry;
/// TomoJAX's own files use /entry): a group, reached through any links, whose
/// `definition` field or attribute is NXtomo. Among those and the
/// groups below them, the one holding a detector; None when there is none or more
/// than one.
fn nxtomo_entry(file: &hdf5::File) -> Result<Option<String>, String> {
    let declares = |group: &hdf5::Group| -> Result<bool, String> {
        let label = if group.link_exists("definition") {
            match group.dataset("definition") {
                Ok(field) => text(&field, "definition")?,
                Err(_) => return Ok(false),
            }
        } else {
            string_attribute(group, "definition")?.unwrap_or_default()
        };
        Ok(label.trim() == "NXtomo")
    };
    let mut definitions = Vec::new();
    let mut containers = Vec::new();
    let mut seen = Vec::new();
    let root = file.group("/").map_err(|error| error.to_string())?;
    let mut pending = vec![(String::new(), root, false)];
    while let Some((path, group, inherited)) = pending.pop() {
        let declared = declares(&group)?;
        let scope = inherited || declared;
        let info = group.loc_info().map_err(|error| error.to_string())?;
        // Object identity across files: soft and external links can form cycles.
        if seen.contains(&(info.fileno, info.token, scope)) {
            continue;
        }
        seen.push((info.fileno, info.token, scope));
        if declared {
            definitions.push(path.clone());
        }
        if scope && group.link_exists("instrument/detector") {
            containers.push(path.clone());
        }
        for name in group.member_names().map_err(|error| error.to_string())? {
            // Opened by name, so an external link resolves into its own file.
            if let Ok(child) = group.group(&name) {
                pending.push((format!("{path}/{name}"), child, scope));
            }
        }
    }
    let mut found = if containers.is_empty() {
        definitions
    } else {
        containers
    };
    Ok((found.len() == 1).then(|| found.remove(0)))
}

fn is_rc6(file: &hdf5::File) -> Result<bool, String> {
    let legacy_json = [
        ("/entry", "grid_meta_json"),
        ("/entry/geometry", "geometry_meta_json"),
        ("/entry/instrument/detector", "detector_meta_json"),
    ]
    .iter()
    .any(|(path, name)| {
        file.group(path)
            .ok()
            .and_then(|g| g.attr_names().ok())
            .is_some_and(|names| names.iter().any(|a| a == name))
    });
    let bare_rc6 = if file.link_exists("/entry/data/projections")
        && let Ok(geometry) = file.group("/entry/geometry")
    {
        let entry = file.group("/entry").map_err(|e| e.to_string())?;
        string_attribute(&geometry, "NX_class")?.as_deref() == Some("NXcollection")
            && string_attribute(&geometry, "type")?.is_some()
            && string_attribute(&entry, "definition")?.as_deref() == Some("NXtomo")
    } else {
        false
    };
    Ok(file.link_exists("/entry/processing/tomojax") || legacy_json || bare_rc6)
}

/// Recognize supported native files before a caller interprets an acquisition layout.
pub fn native_format(file: &hdf5::File) -> Result<bool, String> {
    if !file
        .attr_names()
        .map_err(|e| e.to_string())?
        .iter()
        .any(|name| name == "tomojax_format")
    {
        if is_rc6(file)? {
            return Err("Re-save this rc6 file with TomoJAX 1.0.0rc7 (install with `python -m pip install \"tomojax[cpu]==1.0.0rc7\"`): tj.save(new, tj.load(old)) for a scan, or use tj.load_reconstruction, tj.load_alignment or tj.load_simulation for that result.".into());
        }
        return Ok(false);
    }
    let number = file
        .attr("tomojax_format")
        .and_then(|a| a.read_scalar::<u32>())
        .map_err(|e| e.to_string())?;
    if number != 2 {
        let version =
            string_attribute(file, "creator_version")?.unwrap_or_else(|| "unknown".into());
        return Err(format!(
            "Unsupported TomoJAX format {number} written by {version}"
        ));
    }
    Ok(true)
}

fn native_metadata(file: &hdf5::File, path: &Path) -> Result<Option<HeaderMetadata>, String> {
    use hdf5::types::VarLenUnicode;

    if !native_format(file)? {
        return Ok(None);
    }
    let encoded = file
        .dataset("/entry/record")
        .and_then(|r| r.read_scalar::<VarLenUnicode>())
        .map_err(|e| e.to_string())?;
    let record: Value = serde_json::from_str(encoded.as_str()).map_err(|e| e.to_string())?;
    if record["format"].as_u64() != Some(2) {
        return Err("Inconsistent format in /entry/record".into());
    }
    let contents = record["arrays"].as_object().ok_or("Missing saved arrays")?;
    let mut arrays = BTreeMap::new();
    for (name, value) in contents {
        if !matches!(
            name.as_str(),
            "projections" | "volume" | "image" | "phantom" | "mask"
        ) {
            continue;
        }
        let mut array = value.clone();
        let dataset = value["path"].as_str().ok_or("Missing saved member path")?;
        array["storage"] = json!({"type":"hdf5","path":path,"dataset":dataset});
        for axis in array["axes"].as_array_mut().ok_or("Missing saved axes")? {
            if let Some(table) = axis.pointer("/coordinates/table").and_then(Value::as_str) {
                let coordinates = record["tables"][table]["path"]
                    .as_str()
                    .ok_or("Missing coordinate table")?;
                axis["coordinates"] = json!({"type":"coordinates","values":{"type":"hdf5","path":path,"dataset":coordinates}});
            }
        }
        arrays.insert(
            name.clone(),
            serde_json::from_value(array).map_err(|e| format!("{name}: {e}"))?,
        );
    }
    let volume = ["volume", "image", "phantom"]
        .iter()
        .find_map(|name| contents.get(*name));
    let projection = &record["arrays"]["projections"];
    let angle = &record["tables"]["angles"];
    let num_angles = angle["path"]
        .as_str()
        .map(|name| {
            file.dataset(name)
                .map(|d| d.size())
                .map_err(|e| e.to_string())
        })
        .transpose()?;
    let detector = &record["geometry"]["detector"];
    let report = json!({
        "source": "header_only", "input_path": path, "record": record, "contents": contents,
        "projection": {"found": projection.is_object(), "shape": projection["shape"],
                       "dtype": projection["dtype"], "path": projection["path"], "n_views": projection["shape"][0]},
        "angles": {"found": angle.is_object(), "count": num_angles, "path": angle["path"],
                   "units": num_angles.map(|_| "deg")},
        "detector_metadata": {"found": detector.is_object(), "nu": detector["nu"], "nv": detector["nv"],
                              "du": detector["du"], "dv": detector["dv"], "center": detector["det_center"]},
        "alignment": {"found": record["tables"]["poses"].is_object() || record["tables"]["angle_offset"].is_object(),
                      "params_found": record["tables"]["poses"].is_object()},
        "volume": {"found": volume.is_some(), "shape": volume.map(|v| &v["shape"]), "axes": "zyx",
                   "method": record.pointer("/geometry/meta/reconstruction_method")},
        "geometry": {"type": record.pointer("/geometry/type")}
    });
    Ok(Some((suggested_kind(&report).into(), arrays, report)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn quick_metadata_does_not_read_external_projection_pixels_or_claim_an_audit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("scan.nxs");
        let unavailable = directory.path().join("offline-projection-pixels.raw");
        {
            let file = hdf5::File::create(&path).unwrap();
            let detector = file
                .create_group("entry")
                .unwrap()
                .create_group("instrument")
                .unwrap()
                .create_group("detector")
                .unwrap();
            detector
                .new_dataset::<f32>()
                .shape((2, 3, 4))
                .external(unavailable.to_str().unwrap(), 0, 96)
                .create("data")
                .unwrap();
        }
        assert!(!unavailable.exists());
        let (kind, arrays, report) = quick_metadata(&path).unwrap().unwrap();
        assert_eq!(kind, "frames");
        assert_eq!(arrays["projections"].shape, [2, 3, 4]);
        assert_eq!(report["source"], "header_only");
        assert!(report.get("valid").is_none());
        assert!(report["projection"].get("stats").is_none());
        assert!(!unavailable.exists());
    }

    #[test]
    fn a_beamline_master_is_read_from_its_nxtomo_entry_through_external_links() {
        // As GDA writes a DIAD scan: an NXsubentry declaring NXtomo below /entry,
        // its detector frames in another file behind an external link.
        use hdf5::types::VarLenUnicode;
        let directory = tempfile::tempdir().unwrap();
        let frames = hdf5::File::create(directory.path().join("imaging.h5")).unwrap();
        frames
            .create_group("entry")
            .unwrap()
            .new_dataset::<u16>()
            .shape((4, 2, 3))
            .create("data")
            .unwrap();
        drop(frames);
        let path = directory.path().join("scan.nxs");
        let file = hdf5::File::create(&path).unwrap();
        let entry = file.create_group("entry").unwrap();
        entry.create_group("instrument").unwrap();
        let tomo = entry.create_group("tomo_entry").unwrap();
        tomo.new_dataset::<VarLenUnicode>()
            .shape(())
            .create("definition")
            .unwrap()
            .write_scalar(&"NXtomo".parse::<VarLenUnicode>().unwrap())
            .unwrap();
        let detector = tomo
            .create_group("instrument")
            .unwrap()
            .create_group("detector")
            .unwrap();
        detector
            .link_external("imaging.h5", "/entry/data", "data")
            .unwrap();
        detector
            .new_dataset_builder()
            .with_data(&[1i32, 2, 0, 0])
            .create("image_key")
            .unwrap();
        drop((detector, tomo, entry, file));

        let (kind, _, report) = quick_metadata(&path).unwrap().unwrap();
        assert_eq!(kind, "frames");
        assert_eq!(
            report["projection"]["path"],
            "/entry/tomo_entry/instrument/detector/data"
        );
        assert_eq!(report["projection"]["shape"], json!([4, 2, 3]));
        assert_eq!(
            report["flats_darks"]["image_key_path"],
            "/entry/tomo_entry/instrument/detector/image_key"
        );
        assert_eq!(report["flats_darks"]["sample_count"], 2);
    }

    #[test]
    fn fixed_length_string_attributes_are_read() {
        use hdf5::types::FixedAscii;
        let directory = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(directory.path().join("angles.h5")).unwrap();
        let angles = file
            .new_dataset_builder()
            .with_data(&[0.0f32, 1.])
            .create("angles")
            .unwrap();
        angles
            .new_attr::<FixedAscii<6>>()
            .shape(())
            .create("units")
            .unwrap()
            .write_scalar(&FixedAscii::<6>::from_ascii("degree").unwrap())
            .unwrap();
        assert_eq!(
            string_attribute(&angles, "units").unwrap().as_deref(),
            Some("degree")
        );
    }

    #[test]
    fn a_scan_with_a_pose_table_is_a_scan_and_an_alignment_needs_its_record() {
        let posed = json!({"geometry":{"meta_keys":["corrections"]},
            "alignment":{"found":true,"params_found":true}});
        assert_eq!(suggested_kind(&posed), "scan");
        let aligned = json!({"geometry":{"meta_keys":["alignment_info"]},
            "alignment":{"found":true,"params_found":true}});
        assert_eq!(suggested_kind(&aligned), "alignment");
    }

    #[test]
    fn header_import_does_not_show_flats_and_darks_as_views() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("raw.nxs");
        {
            let file = hdf5::File::create(&path).unwrap();
            let detector = file
                .create_group("entry")
                .unwrap()
                .create_group("instrument")
                .unwrap()
                .create_group("detector")
                .unwrap();
            detector
                .new_dataset::<u16>()
                .shape((4, 2, 3))
                .create("data")
                .unwrap();
            detector
                .new_dataset_builder()
                .with_data(&[0i32, 1, 2, 0])
                .create("image_key")
                .unwrap();
        }
        let (kind, arrays, report) = quick_metadata(&path).unwrap().unwrap();
        assert_eq!(kind, "frames");
        assert_eq!(report["flats_darks"]["sample_count"], 2);
        assert!(!arrays.contains_key("projections"));
        assert_eq!(arrays["counts"].axes[0].name, "frame");
        assert_eq!(arrays["counts"].domain, "counts");

        // ESRF files also mark return images with -1 in image_key_control; the
        // image keys still give the flats and darks.
        {
            let file = hdf5::File::open_rw(&path).unwrap();
            let detector = file.group("/entry/instrument/detector").unwrap();
            detector
                .new_dataset_builder()
                .with_data(&[0i32, 1, 2, -1])
                .create("image_key_control")
                .unwrap();
        }
        let (kind, arrays, report) = quick_metadata(&path).unwrap().unwrap();
        assert_eq!(kind, "frames");
        assert_eq!(report["flats_darks"]["sample_count"], 1);
        assert!(!arrays.contains_key("projections"));
        assert_eq!(arrays["counts"].axes[0].name, "frame");
        assert_eq!(arrays["counts"].domain, "counts");

        // Keys stored elsewhere (an NXdata group) are found as the worker finds
        // them: the file's only per-frame `image_key`.
        {
            let file = hdf5::File::open_rw(&path).unwrap();
            let detector = file.group("/entry/instrument/detector").unwrap();
            detector.unlink("image_key").unwrap();
            detector.unlink("image_key_control").unwrap();
            file.create_group("/entry/data")
                .unwrap()
                .new_dataset_builder()
                .with_data(&[1i32, 0, 0, 2])
                .create("image_key")
                .unwrap();
        }
        let (kind, arrays, report) = quick_metadata(&path).unwrap().unwrap();
        assert_eq!(kind, "frames");
        assert_eq!(report["flats_darks"]["sample_count"], 2);
        assert_eq!(report["flats_darks"]["first_sample"], 1);
        assert!(!arrays.contains_key("projections"));
        assert_eq!(arrays["counts"].axes[0].name, "frame");
        assert_eq!(arrays["counts"].domain, "counts");
    }

    #[test]
    fn saved_feature_registers_its_mask_for_the_worker() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tests/data/formats/format2-feature.nxs");
        let (kind, arrays, _) = quick_metadata(&path).unwrap().unwrap();
        assert_eq!(kind, "feature");
        assert_eq!(arrays["mask"].shape, [2, 3, 4]);
        assert_eq!(arrays["mask"].domain, "mask");
        assert!(
            matches!(&arrays["mask"].storage, Storage::Hdf5 {dataset, ..} if dataset == "/entry/masks/grain/data")
        );
    }

    #[test]
    fn a_format_one_file_gives_resave_instructions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("legacy.nxs");
        {
            let file = hdf5::File::create(&path).unwrap();
            file.create_group("entry").unwrap();
            file.create_group("/entry/processing")
                .unwrap()
                .create_group("tomojax")
                .unwrap()
                .new_dataset::<f32>()
                .shape((4, 3, 4))
                .create("volume")
                .unwrap();
        }
        let error = inspection_arrays(
            &json!({"volume":{"found":true},"projection":{"found":false}}),
            &path,
        )
        .unwrap_err();
        assert!(
            error.contains("python -m pip install \"tomojax[cpu]==1.0.0rc7\""),
            "{error}"
        );
        assert!(error.contains("tj.load_reconstruction"), "{error}");
    }

    #[test]
    fn existing_alignment_image_uses_its_working_grid_without_loading_voxels() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tests/data/formats/format2-alignment.nxs");
        let arrays = inspection_arrays(
            &json!({"volume":{"found":true},"projection":{"found":false}}),
            &path,
        )
        .unwrap();
        let image = &arrays["image"];
        assert_eq!(image.shape, [2, 3, 4]);
        assert_eq!(image.dtype, "float32");
        assert_eq!(image.unit, "1/mm");
        let expected = [("z", 3., 0.3), ("y", 2., 0.2), ("x", -1., 0.1)];
        for (axis, (name, origin, spacing)) in image.axes.iter().zip(expected) {
            assert_eq!(axis.name, name);
            assert_eq!(axis.unit, "mm");
            assert_eq!(axis.coordinates, Coordinates::Uniform { origin, spacing });
        }
        let (kind, quick, report) = quick_metadata(&path).unwrap().unwrap();
        assert_eq!(kind, "alignment");
        assert_eq!(quick["image"], *image);
        assert_eq!(report["source"], "header_only");
        assert!(report.get("valid").is_none());
    }

    #[test]
    fn direct_measurements_do_not_claim_loader_corrections_or_unrecorded_units() {
        use hdf5::types::VarLenUnicode;
        use std::str::FromStr;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("scan.h5");
        let file = hdf5::File::create(&path).unwrap();
        file.new_dataset::<u16>()
            .shape((3, 2, 4))
            .create("raw")
            .unwrap();
        file.new_dataset::<u16>()
            .shape((5, 2, 4))
            .create("with_flats")
            .unwrap();
        file.new_dataset_builder()
            .with_data(&[0.0f32, 0.5, 1.0])
            .create("theta")
            .unwrap()
            .new_attr::<VarLenUnicode>()
            .shape(())
            .create("units")
            .unwrap()
            .write_scalar(&VarLenUnicode::from_str("rad").unwrap())
            .unwrap();
        file.new_dataset_builder()
            .with_data(&[0.0f32, 0.5, 1.0])
            .create("unlabeled")
            .unwrap();
        file.new_dataset::<u16>()
            .shape((3, 4))
            .create("line")
            .unwrap();
        drop(file);
        // The worker reports converted degrees; the stored angles are radians.
        let report = |data: &str, angles: &str| {
            json!({"projection":{"found":true,"path":data,"shape":[3,2,4],"dtype":"uint16"},
            "angles":{"path":angles,"units":"degree","count":3},
            "detector_metadata":{"du":0.1,"dv":0.4,"center":[0.3,-0.2]}})
        };
        // A sample-only report cannot directly index the longer stored detector stack.
        assert!(
            projection_arrays(&report("/with_flats", "/theta"), &path)
                .unwrap()
                .is_empty()
        );
        let mut whole = report("/raw", "/theta");
        whole["flats_darks"] = json!({"sample_count": 2});
        let mixed = projection_arrays(&whole, &path).unwrap();
        assert_eq!(mixed["counts"].shape, [3, 2, 4]);
        assert_eq!(mixed["counts"].axes[0].unit, "index");
        assert!(!mixed.contains_key("projections"));
        // A stored (view, u) line scan registers as (view, 1, u).
        let mut line = report("/line", "/theta");
        line["projection"]["shape"] = json!([3, 1, 4]);
        assert_eq!(
            projection_arrays(&line, &path).unwrap()["projections"].shape,
            [3, 1, 4]
        );
        let unlabeled = projection_arrays(&report("/raw", "/unlabeled"), &path).unwrap();
        assert_eq!(unlabeled["projections"].axes[0].unit, "index");
        let arrays = projection_arrays(&report("/raw", "/theta"), &path).unwrap();
        let array = &arrays["projections"];
        assert_eq!(array.domain, "stored_measurements");
        assert_eq!(array.axes[0].unit, "radian");
        assert_eq!(array.axes[1].unit, "scan_length");
        assert_eq!(
            array.axes[1].coordinates,
            Coordinates::Uniform {
                origin: -0.4,
                spacing: 0.4
            }
        );
        if let Coordinates::Uniform { origin, spacing } = array.axes[2].coordinates {
            assert!((origin - 0.15).abs() < 1e-12);
            assert_eq!(spacing, 0.1);
        } else {
            panic!("expected detector pixel coordinates");
        }
    }

    #[test]
    fn simulation_truth_is_not_misclassified_as_a_reconstruction() {
        let mut report = json!({"volume":{"found":true,"method":null},"geometry":{"meta_keys":["simulation_info","volume_role"]}});
        assert_eq!(suggested_kind(&report), "simulation");
        report["volume"]["method"] = json!("fourier");
        assert_eq!(suggested_kind(&report), "reconstruction");
        assert!(
            projection_arrays(&report, Path::new("volume.nxs"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn corrected_scan_retains_its_acquisition_history_without_becoming_raw_again() {
        let report = json!({"volume":{"found":false},"geometry":{"meta_keys":["acquisition","corrections"]}});
        assert_eq!(suggested_kind(&report), "scan");
    }
}
