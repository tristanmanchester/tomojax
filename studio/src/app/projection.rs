//! Which selected inputs a forward or back projection reads.
//!
//! A projection job takes its geometry from one input and its volume (forward)
//! or projections (back) from another, or from the same one. `ProjectionRoles`
//! makes that choice once, from the opened inputs; preflight, the worker command
//! and the result's parent all read it, and the form shows it.

use serde_json::{Map, Value, json};

use crate::protocol::{ArrayDescriptor, Coordinates, Product, Storage};

/// The roles of a projection job's inputs, as indices into its inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionRoles {
    /// "volume" for forward projection, "projections" for back projection.
    pub key: &'static str,
    /// The input whose scan supplies the geometry.
    pub geometry: usize,
    /// The input holding the array the job reads; `None` when the job names the
    /// array by storage reference.
    pub array: Option<usize>,
    storage: Option<Storage>,
    grid: Option<Value>,
}

impl ProjectionRoles {
    /// Chooses the roles for `operation` ("project" or "backproject") from the
    /// opened inputs, in job input order. `parameters` are the job's own; an
    /// array given there by reference is used as given.
    ///
    /// A selected array result (a saved volume or forward projection) supplies
    /// the array; otherwise a reconstructed or aligned volume before a
    /// simulation's phantom, and a scan's projections before a simulation's.
    /// Back projection takes its geometry from the scan whose projections it
    /// reads; forward projection from a selected scan other than the volume's
    /// holder, so a reconstruction can be projected into another scan.
    pub fn choose(
        operation: &str,
        products: &[&Product],
        parameters: &Value,
    ) -> Result<Self, String> {
        let key = if operation == "project" {
            "volume"
        } else {
            "projections"
        };
        let referenced = parameters.get(key).is_some_and(|value| !value.is_null());
        let array = if referenced {
            None
        } else {
            Some(array_source(products, key).ok_or_else(|| {
                format!("Select an input containing {key}, or provide its storage reference")
            })?)
        };
        let scans = || {
            products
                .iter()
                .enumerate()
                .filter(|(_, product)| product.scan_source().is_some())
                .map(|(index, _)| index)
        };
        let geometry = match &array {
            Some(array) if !array.registered && key == "projections" => Some(array.index),
            Some(array) if !array.registered => scans()
                .find(|&index| index != array.index)
                .or_else(|| scans().find(|&index| index == array.index)),
            _ => scans().next(),
        }
        .ok_or("Select a scan to supply the geometry")?;
        let descriptor = array.as_ref().and_then(|array| array.descriptor.as_ref());
        let grid = (key == "volume" && parameters.get("grid").is_none_or(Value::is_null))
            .then(|| descriptor.and_then(registered_grid))
            .flatten();
        Ok(Self {
            key,
            geometry,
            array: array.as_ref().map(|array| array.index),
            storage: array.map(|array| array.storage),
            grid,
        })
    }

    /// Puts the chosen array, and a forward projection's volume grid, into the
    /// worker command.
    pub fn apply(&self, operation: &mut Map<String, Value>) -> Result<(), String> {
        if let Some(storage) = &self.storage {
            operation.insert(
                self.key.into(),
                serde_json::to_value(storage).map_err(|e| e.to_string())?,
            );
        }
        if let Some(grid) = &self.grid {
            operation.insert("grid".into(), grid.clone());
        }
        Ok(())
    }

    /// The input a back projection's volume belongs to: the one whose
    /// projections it read, which Explore shows beside it. Projections given by
    /// reference belong to no input.
    pub fn parent(&self) -> Option<usize> {
        if self.key == "projections" {
            self.array
        } else {
            Some(self.geometry)
        }
    }

    /// "Volume: A · geometry: B", from the inputs' names.
    pub fn describe(&self, names: &[String]) -> String {
        let name = |index: usize| names.get(index).map_or("?", String::as_str);
        let array = self.array.map_or("given by reference", name);
        let what = if self.key == "volume" {
            "Volume"
        } else {
            "Projections"
        };
        format!("{what}: {array} · geometry: {}", name(self.geometry))
    }
}

/// Where the array comes from.
#[derive(Debug)]
struct ArraySource {
    index: usize,
    /// A selected array result rather than a member of a scan or reconstruction.
    registered: bool,
    storage: Storage,
    descriptor: Option<ArrayDescriptor>,
}

fn array_source(products: &[&Product], key: &str) -> Option<ArraySource> {
    let registered = products.iter().enumerate().find_map(|(index, product)| {
        let array = match product {
            Product::Volume { volume } if key == "volume" => volume,
            Product::Projections { projections } if key == "projections" => projections,
            _ => return None,
        };
        Some(ArraySource {
            index,
            registered: true,
            storage: array.storage.clone(),
            descriptor: Some(array.clone()),
        })
    });
    registered.or_else(|| {
        products
            .iter()
            .enumerate()
            .filter_map(|(index, product)| {
                let members = product.members();
                let (rank, member, descriptor) = match product {
                    Product::Scan { projections, .. } if key == "projections" => {
                        (0, "projections", Some(projections))
                    }
                    _ if key == "projections" => (1, "projections", None),
                    Product::Reconstruction { volume, .. } => (0, "volume", Some(volume)),
                    Product::Alignment { image, .. } => (1, "image", image.as_ref()),
                    Product::Simulation { phantom, .. } => (2, "phantom", Some(phantom)),
                    _ => (3, "volume", None),
                };
                Some((
                    rank,
                    ArraySource {
                        index,
                        registered: false,
                        storage: members.get(member)?.clone(),
                        descriptor: descriptor.cloned(),
                    },
                ))
            })
            .min_by_key(|(rank, source)| (*rank, source.index))
            .map(|(_, source)| source)
    })
}

/// The grid of a volume registered with uniform (z, y, x) voxel-center axes.
fn registered_grid(volume: &ArrayDescriptor) -> Option<Value> {
    let names: Vec<_> = volume.axes.iter().map(|axis| axis.name.as_str()).collect();
    if names != ["z", "y", "x"] || volume.shape.len() != 3 {
        return None;
    }
    let mut uniform = volume.axes.iter().map(|axis| match axis.coordinates {
        Coordinates::Uniform { origin, spacing } => Some((origin, spacing)),
        Coordinates::Coordinates { .. } => None,
    });
    let (z, y, x) = (uniform.next()??, uniform.next()??, uniform.next()??);
    Some(json!({
        "nx": volume.shape[2], "ny": volume.shape[1], "nz": volume.shape[0],
        "vx": x.1, "vy": y.1, "vz": z.1,
        "vol_origin": [x.0, y.0, z.0],
    }))
}
