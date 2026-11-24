//! Interpret the public inspection report and the storage layout of committed NXtomo.
//!
//! Inspection intentionally omits some calibration. Never promote guessed voxel
//! sizes or uncorrected file pixels into a registered reconstruction or scan.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Value, json};

use crate::protocol::{ArrayDescriptor, Axis, Coordinates, Grid, Product, Storage};

pub const PROJECTIONS: &str = "/entry/instrument/detector/data";
pub const VOLUME: &str = "/entry/processing/tomojax/volume";
pub const ANGLES: &str = "/entry/sample/transformations/rotation_angle";

/// Suggested input kind, registered file arrays and the metadata-only report.
pub type HeaderMetadata = (String, BTreeMap<String, ArrayDescriptor>, Value);

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
    if !file.link_exists("/entry") {
        return Ok(None);
    }
    let projection_path = [PROJECTIONS, "/entry/data/projections", "/entry/projections"]
        .into_iter()
        .find(|name| file.link_exists(name));
    let has_volume = file.link_exists(VOLUME);
    if projection_path.is_none() && !has_volume {
        return Ok(None);
    }
    let entry = file.group("/entry").map_err(|error| error.to_string())?;
    let (geometry, geometry_type) = if file.link_exists("/entry/geometry") {
        let group = file
            .group("/entry/geometry")
            .map_err(|error| error.to_string())?;
        (
            json_attribute(&group, "geometry_meta_json")?,
            string_attribute(&group, "type")?,
        )
    } else {
        (Value::Null, None)
    };
    let detector = if file.link_exists("/entry/instrument/detector") {
        json_attribute(
            &file
                .group("/entry/instrument/detector")
                .map_err(|error| error.to_string())?,
            "detector_meta_json",
        )?
    } else {
        Value::Null
    };
    let mut projection = json!({"found":false,"path":null,"shape":null,"dtype":null});
    if let Some(dataset) = projection_path {
        let data = file.dataset(dataset).map_err(|error| error.to_string())?;
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
    let angle_path = ["/entry/sample/rotation_angle", ANGLES]
        .into_iter()
        .find(|name| file.link_exists(name));
    let angles = if let Some(dataset) = angle_path {
        let values = file.dataset(dataset).map_err(|error| error.to_string())?;
        let shape = values.shape();
        if shape.len() != 1 {
            return Ok(None);
        }
        json!({"found":true,"path":dataset,"count":shape[0],"units":string_attribute(&values,"units")?})
    } else {
        json!({"found":false,"path":null,"count":null,"units":null})
    };
    let volume = if has_volume {
        let data = file.dataset(VOLUME).map_err(|error| error.to_string())?;
        let group = file
            .group("/entry/processing/tomojax")
            .map_err(|error| error.to_string())?;
        json!({"found":true,"shape":data.shape(),"axes":string_attribute(&group,"volume_axes_order")?.unwrap_or_else(||"zyx".into()),"method":geometry.get("reconstruction_method")})
    } else {
        json!({"found":false,"shape":null,"axes":null,"method":null})
    };
    let image_key = "/entry/instrument/detector/image_key";
    let key_found = file.link_exists(image_key);
    // NXtomo image keys mark sample (0), flat (1) and dark (2) frames, and
    // `image_key_control` marks return and alignment images (-1) to leave out, as
    // the worker does; the keys are one small integer per frame.
    let read_keys = |name: &str| -> Result<Option<Vec<i64>>, String> {
        if !file.link_exists(name) {
            return Ok(None);
        }
        file.dataset(name)
            .and_then(|keys| keys.read_raw::<i64>())
            .map(Some)
            .map_err(|error| error.to_string())
    };
    let control = read_keys("/entry/instrument/detector/image_key_control")?;
    let sample_count = read_keys(image_key)?.map(|keys| {
        keys.iter()
            .enumerate()
            .filter(|&(i, &key)| {
                key == 0
                    && control
                        .as_ref()
                        .is_none_or(|control| control.get(i) != Some(&-1))
            })
            .count()
    });
    let flat_found = file.link_exists("/entry/instrument/detector/data_white");
    let dark_found = file.link_exists("/entry/instrument/detector/data_dark");
    let tomojax_grid = entry
        .attr_names()
        .map_err(|error| error.to_string())?
        .iter()
        .any(|name| name == "grid_meta_json");
    let report = json!({"source":"header_only","input_path":path,
        "projection":projection,"angles":angles,"volume":volume,
        "geometry":{"found":geometry_type.is_some(),"type":geometry_type,"meta_found":geometry.is_object(),"meta_keys":geometry.as_object().map(|meta|meta.keys().collect::<Vec<_>>()).unwrap_or_default()},
        "detector_metadata":{"found":detector.is_object(),"nu":detector.get("nu"),"nv":detector.get("nv"),"du":detector.get("du"),"dv":detector.get("dv"),"center":detector.get("det_center")},
        "flats_darks":{"image_key_found":key_found,"image_key_path":if key_found{Some(image_key)}else{None},
            "sample_count":sample_count,
            "flats_present":if flat_found{Some(true)}else{None},"darks_present":if dark_found{Some(true)}else{None}},
        "alignment":{"found":file.link_exists("/entry/processing/tomojax/align"),"params_found":file.link_exists("/entry/processing/tomojax/align/thetas")}});
    let mut kind = suggested_kind(&report).to_owned();
    let interleaved = sample_count
        .is_some_and(|samples| report["projection"]["n_views"].as_u64() != Some(samples as u64));
    if kind == "scan" && (interleaved || !(tomojax_grid && detector.is_object())) {
        kind = "frames".into();
    }
    let mut arrays = projection_arrays(&report, path)?;
    let length_unit = geometry
        .get("length_unit")
        .or_else(|| geometry.pointer("/acquisition/length_unit"))
        .and_then(Value::as_str)
        .filter(|unit| !unit.is_empty())
        .unwrap_or("scan_length");
    if let Some(array) = arrays.get_mut("projections") {
        for axis in &mut array.axes[1..] {
            if axis.unit == "scan_length" {
                axis.unit = length_unit.into();
            }
        }
        // Raw HDF5 may interleave sample/calibration frames whose angle entries
        // are undefined. Until the loader selects samples, show stored indices.
        if kind == "frames" && key_found {
            array.axes[0].unit = "index".into();
            array.axes[0].coordinates = Coordinates::Uniform {
                origin: 0.,
                spacing: 1.,
            };
        }
    }
    if has_volume {
        let (name, array) = stored_volume(&file, path, &geometry, length_unit)?;
        arrays.insert(name, array);
    }
    Ok(Some((kind, arrays, report)))
}

/// A suggested loader, still editable when an acquisition's domain is ambiguous.
/// Whether `path` names an HDF5 file (NXtomo and other HDF5 layouts).
pub fn is_hdf5(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        matches!(
            e.to_ascii_lowercase().as_str(),
            "nx" | "nxs" | "h5" | "hdf" | "hdf5"
        )
    })
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

pub fn suggested_kind(report: &Value) -> &'static str {
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
    } else if has("alignment_info")
        || report
            .pointer("/alignment/params_found")
            .and_then(Value::as_bool)
            == Some(true)
    {
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
    let geometry = if file.link_exists("/entry/geometry") {
        let group = file
            .group("/entry/geometry")
            .map_err(|error| error.to_string())?;
        json_attribute(&group, "geometry_meta_json")?
    } else {
        Value::Null
    };
    let length_unit = geometry
        .get("length_unit")
        .or_else(|| geometry.pointer("/acquisition/length_unit"))
        .and_then(Value::as_str)
        .filter(|unit| !unit.is_empty())
        .unwrap_or("scan_length");
    let mut arrays = projection_arrays(report, path)?;
    if let Some(projections) = arrays.get_mut("projections") {
        for axis in &mut projections.axes[1..] {
            if axis.unit == "scan_length" {
                axis.unit = length_unit.to_owned();
            }
        }
    }
    if file.link_exists(VOLUME) {
        let (name, volume) = stored_volume(&file, path, &geometry, length_unit)?;
        arrays.insert(name, volume);
    }
    Ok(arrays)
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
    // Registered for direct viewing only when every stored frame is a sample
    // view; frames interleaved with flats and darks become viewable once the scan
    // is opened and calibrated. A stored (view, u) line scan registers as
    // (view, 1, u).
    let samples = report
        .pointer("/flats_darks/sample_count")
        .and_then(Value::as_u64);
    let stored = hdf5::File::open(path).map_err(|error| error.to_string())?;
    let squeezed = |dims: &[usize]| dims.iter().copied().filter(|&n| n != 1).collect::<Vec<_>>();
    let stored_shape = stored.dataset(dataset).map(|d| d.shape()).ok();
    if samples.is_some_and(|n| n != shape[0] as u64)
        || stored_shape.as_deref().map(squeezed) != Some(squeezed(&shape))
    {
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
            if dataset.starts_with('/') && angles["count"].as_u64() == Some(shape[0] as u64) =>
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
            name: "view".into(),
            frame: "acquisition".into(),
            unit: "index".into(),
            coordinates: Coordinates::Uniform {
                origin: 0.0,
                spacing: 1.0,
            },
        },
    };
    let detector = &report["detector_metadata"];
    let centre = detector["center"].as_array();
    let mut axes = vec![view];
    for (name, index, pitch) in [("v", 1, "dv"), ("u", 0, "du")] {
        let dimension = if name == "v" { shape[1] } else { shape[2] };
        let calibrated = detector[pitch]
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.0)
            .zip(
                centre
                    .and_then(|c| c.get(index))
                    .and_then(Value::as_f64)
                    .filter(|v| v.is_finite()),
            );
        let (unit, origin, spacing) = match calibrated {
            Some((spacing, centre)) => (
                "scan_length",
                centre - (dimension - 1) as f64 * spacing / 2.0,
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
        "projections".into(),
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
            domain: "stored_measurements".into(),
            fixed: Vec::new(),
            columns: Vec::new(),
        },
    );
    Ok(arrays)
}

fn string_attribute(group: &hdf5::Location, name: &str) -> Result<Option<String>, String> {
    use hdf5::types::{FixedAscii, FixedUnicode, VarLenAscii, VarLenUnicode};
    if !group
        .attr_names()
        .map_err(|error| error.to_string())?
        .iter()
        .any(|key| key == name)
    {
        return Ok(None);
    }
    let attribute = group.attr(name).map_err(|error| error.to_string())?;
    // NeXus writers use variable- and fixed-length strings alike.
    let text = if let Ok(value) = attribute.read_scalar::<VarLenUnicode>() {
        value.as_str().to_owned()
    } else if let Ok(value) = attribute.read_scalar::<VarLenAscii>() {
        value.as_str().to_owned()
    } else if let Ok(value) = attribute.read_scalar::<FixedUnicode<4096>>() {
        value.as_str().to_owned()
    } else {
        attribute
            .read_scalar::<FixedAscii<4096>>()
            .map_err(|error| format!("Cannot decode {name}: {error}"))?
            .as_str()
            .to_owned()
    };
    if text.len() > 1024 * 1024 {
        return Err(format!("The {name} metadata exceed 1 MiB"));
    }
    Ok(Some(text))
}

fn json_attribute(group: &hdf5::Group, name: &str) -> Result<Value, String> {
    string_attribute(group, name)?
        .map(|text| {
            serde_json::from_str(&text).map_err(|error| format!("Malformed {name}: {error}"))
        })
        .transpose()
        .map(|value| value.unwrap_or(Value::Null))
}

fn stored_volume(
    file: &hdf5::File,
    path: &Path,
    geometry: &Value,
    length_unit: &str,
) -> Result<(String, ArrayDescriptor), String> {
    use hdf5::types::TypeDescriptor;
    let volume = file.dataset(VOLUME).map_err(|error| error.to_string())?;
    let shape = volume.shape();
    if shape.len() != 3 || shape.contains(&0) {
        return Err("A stored volume must have three nonempty dimensions".into());
    }
    let dtype = volume
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .map_err(|error| error.to_string())?;
    if !matches!(
        dtype,
        TypeDescriptor::Float(_) | TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_)
    ) {
        return Err(format!(
            "The stored volume's {dtype} datatype is not a scalar image"
        ));
    }
    let processing = file
        .group("/entry/processing/tomojax")
        .map_err(|error| error.to_string())?;
    let order = string_attribute(&processing, "volume_axes_order")?.unwrap_or_else(|| "zyx".into());
    let mut names: Vec<_> = order.chars().collect();
    names.sort_unstable();
    if names != ['x', 'y', 'z'] {
        return Err("The stored volume axis order must contain x, y and z exactly once".into());
    }
    let entry = file.group("/entry").map_err(|error| error.to_string())?;
    let recorded_grid = json_attribute(&entry, "grid_meta_json")?;
    let grid_value = if geometry.get("reconstruction_method").is_none() {
        geometry
            .pointer("/alignment_info/working_grid")
            .filter(|value| value.is_object())
            .unwrap_or(&recorded_grid)
    } else {
        &recorded_grid
    };
    let grid = if grid_value.is_null() {
        None
    } else {
        Some(
            serde_json::from_value::<Grid>(grid_value.clone())
                .map_err(|error| format!("Invalid stored voxel grid: {error}"))?,
        )
    };
    let axes = if let Some(grid) = grid {
        let sizes = [grid.nx, grid.ny, grid.nz];
        let spacing = [grid.vx, grid.vy, grid.vz];
        if sizes.contains(&0) || spacing.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return Err("Stored voxel counts and pitches must be finite and positive".into());
        }
        let centre = grid.vol_center.unwrap_or([0.; 3]);
        let origin = grid.vol_origin.unwrap_or_else(|| {
            std::array::from_fn(|i| centre[i] - (sizes[i] - 1) as f64 * spacing[i] / 2.)
        });
        if origin.iter().any(|v| !v.is_finite()) {
            return Err("Stored voxel origin must be finite".into());
        }
        let mut axes = Vec::new();
        for (dimension, name) in order.chars().enumerate() {
            let coordinate = "xyz".find(name).expect("validated axis order");
            if shape[dimension] != sizes[coordinate] {
                return Err("The stored volume shape differs from its recorded voxel grid".into());
            }
            axes.push(Axis {
                name: name.to_string(),
                frame: "volume".into(),
                unit: length_unit.into(),
                coordinates: Coordinates::Uniform {
                    origin: origin[coordinate],
                    spacing: spacing[coordinate],
                },
            });
        }
        axes
    } else {
        order
            .chars()
            .map(|name| Axis {
                name: name.to_string(),
                frame: "stored_volume".into(),
                unit: "index".into(),
                coordinates: Coordinates::Uniform {
                    origin: 0.,
                    spacing: 1.,
                },
            })
            .collect()
    };
    let registered = !grid_value.is_null();
    let role = if geometry.get("volume_role").and_then(Value::as_str) == Some("phantom") {
        "phantom"
    } else if geometry.get("reconstruction_method").is_none()
        && geometry.get("alignment_info").is_some()
    {
        "image"
    } else {
        "volume"
    };
    Ok((
        role.into(),
        ArrayDescriptor {
            storage: Storage::Hdf5 {
                path: path.to_string_lossy().into_owned(),
                dataset: VOLUME.into(),
            },
            shape,
            dtype: dtype.to_string(),
            axes,
            unit: if registered {
                format!("1/{length_unit}")
            } else {
                "unspecified".into()
            },
            domain: "attenuation".into(),
            fixed: Vec::new(),
            columns: Vec::new(),
        },
    ))
}

/// Rebind the registration of a successfully saved product to its committed file.
/// Call only after the official save terminal; this performs no filesystem writes.
pub fn saved_arrays(
    product: &Product,
    path: &Path,
    report: Option<&Value>,
) -> Result<BTreeMap<String, ArrayDescriptor>, String> {
    let mut arrays = BTreeMap::new();
    let registered = match product {
        Product::Scan { projections, .. } => Some(("projections", PROJECTIONS, projections)),
        Product::Reconstruction { volume, .. } => Some(("volume", VOLUME, volume)),
        Product::Simulation { phantom, .. } => Some(("phantom", VOLUME, phantom)),
        Product::Alignment { image, .. } => image.as_ref().map(|array| ("image", VOLUME, array)),
        _ => None,
    };
    let file = path.to_string_lossy().into_owned();
    if let Some((name, dataset, array)) = registered {
        let mut persisted = array.clone();
        persisted.storage = Storage::Hdf5 {
            path: file.clone(),
            dataset: dataset.into(),
        };
        for axis in &mut persisted.axes {
            if let Coordinates::Coordinates {
                values: Storage::Member { member, .. },
            } = &axis.coordinates
            {
                if member != "angles" {
                    return Err(format!("No durable NXtomo coordinate mapping for {member}"));
                }
                axis.coordinates = Coordinates::Coordinates {
                    values: Storage::Hdf5 {
                        path: file.clone(),
                        dataset: ANGLES.into(),
                    },
                };
            }
        }
        arrays.insert(name.into(), persisted);
    }
    if !arrays.contains_key("projections")
        && product.scan_source().is_some()
        && let Some(report) = report
        && let Some(mut projections) = inspection_arrays(report, path)?.remove("projections")
    {
        projections.unit = "1".into();
        projections.domain = "line_integrals".into();
        if let Some(unit) = product
            .primary_array()
            .and_then(|a| {
                a.axes
                    .iter()
                    .find(|axis| matches!(axis.name.as_str(), "z" | "y" | "x"))
            })
            .map(|a| &a.unit)
        {
            for axis in &mut projections.axes[1..] {
                if axis.unit == "scan_length" {
                    axis.unit.clone_from(unit);
                }
            }
        }
        arrays.insert("projections".into(), projections);
    }
    Ok(arrays)
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
    }

    #[test]
    fn existing_alignment_image_uses_its_working_grid_without_loading_voxels() {
        use hdf5::types::VarLenUnicode;
        use std::str::FromStr;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("alignment.nxs");
        {
            let file = hdf5::File::create(&path).unwrap();
            let entry = file.create_group("entry").unwrap();
            let geometry = entry.create_group("geometry").unwrap();
            let processing = entry
                .create_group("processing")
                .unwrap()
                .create_group("tomojax")
                .unwrap();
            let acquisition = json!({"nx":12,"ny":12,"nz":6,"vx":1.,"vy":1.,"vz":1.});
            let meta = json!({"length_unit":"mm","alignment_info":{"working_grid":{
                "nx":4,"ny":3,"nz":2,"vx":0.25,"vy":0.5,"vz":2.,
                "vol_origin":[10.,20.,30.],"vol_center":[0.,0.,0.]}}});
            for (group, key, text) in [
                (&entry, "grid_meta_json", acquisition.to_string()),
                (&geometry, "geometry_meta_json", meta.to_string()),
                (&processing, "volume_axes_order", "zyx".into()),
            ] {
                group
                    .new_attr::<VarLenUnicode>()
                    .shape(())
                    .create(key)
                    .unwrap()
                    .write_scalar(&VarLenUnicode::from_str(&text).unwrap())
                    .unwrap();
            }
            processing
                .new_dataset::<f32>()
                .shape((2, 3, 4))
                .create("volume")
                .unwrap();
        }
        let arrays = inspection_arrays(
            &json!({"volume":{"found":true},"projection":{"found":false}}),
            &path,
        )
        .unwrap();
        let image = &arrays["image"];
        assert_eq!(image.shape, [2, 3, 4]);
        assert_eq!(image.dtype, "float32");
        assert_eq!(image.unit, "1/mm");
        let expected = [("z", 30., 2.), ("y", 20., 0.5), ("x", 10., 0.25)];
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
            .create("unlabelled")
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
        // Sample frames interleaved with flats and darks are not the stored array,
        // whether the report gives the sample stack or the whole stack.
        assert!(
            projection_arrays(&report("/with_flats", "/theta"), &path)
                .unwrap()
                .is_empty()
        );
        let mut whole = report("/raw", "/theta");
        whole["flats_darks"] = json!({"sample_count": 2});
        assert!(projection_arrays(&whole, &path).unwrap().is_empty());
        // A stored (view, u) line scan registers as (view, 1, u).
        let mut line = report("/line", "/theta");
        line["projection"]["shape"] = json!([3, 1, 4]);
        assert_eq!(
            projection_arrays(&line, &path).unwrap()["projections"].shape,
            [3, 1, 4]
        );
        let unlabelled = projection_arrays(&report("/raw", "/unlabelled"), &path).unwrap();
        assert_eq!(unlabelled["projections"].axes[0].unit, "index");
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
    fn saved_product_retains_anisotropic_registration() {
        let array = ArrayDescriptor {
            storage: Storage::Member {
                handle: "h".into(),
                member: "volume".into(),
            },
            shape: vec![2, 3, 4],
            dtype: "<f4".into(),
            unit: "1/mm".into(),
            domain: "attenuation".into(),
            fixed: vec![],
            columns: vec![],
            axes: [("z", 8.0, 2.0), ("y", -3.0, 0.5), ("x", 7.0, -0.25)]
                .map(|(name, origin, spacing)| Axis {
                    name: name.into(),
                    frame: "volume".into(),
                    unit: "mm".into(),
                    coordinates: Coordinates::Uniform { origin, spacing },
                })
                .to_vec(),
        };
        let product = Product::Reconstruction {
            handle: "h".into(),
            volume: array.clone(),
            scan: None,
            method: "fourier".into(),
        };
        let saved = saved_arrays(&product, Path::new("derived/result.nxs"), None).unwrap();
        let volume = &saved["volume"];
        assert_eq!(volume.axes, array.axes);
        assert_eq!(volume.shape, array.shape);
        assert_eq!(volume.unit, "1/mm");
        assert_eq!(
            volume.storage,
            Storage::Hdf5 {
                path: "derived/result.nxs".into(),
                dataset: VOLUME.into()
            }
        );
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
