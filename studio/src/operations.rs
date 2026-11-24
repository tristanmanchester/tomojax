//! User-facing operation vocabulary, shared by projects and the processing forms.
pub struct Operation {
    pub id: &'static str,
    pub title: &'static str,
    pub help: &'static str,
}
pub struct Group {
    pub title: &'static str,
    pub reconstruction: bool,
    pub operations: &'static [Operation],
}
pub const GROUPS: &[Group] = &[
    Group {
        title: "Acquisition",
        reconstruction: false,
        operations: &[
            Operation {
                id: "open",
                title: "Open & calibrate",
                help: "Read a scan or raw acquisition with explicit calibration",
            },
            Operation {
                id: "inspect",
                title: "Acquisition inspection",
                help: "Run the full TomoJAX audit and projection statistics",
            },
            Operation {
                id: "simulate",
                title: "Simulation",
                help: "Generate a registered scan and its ground-truth phantom",
            },
            Operation {
                id: "combine",
                title: "Combined scans",
                help: "Combine selected scans with ordered provenance",
            },
        ],
    },
    Group {
        title: "Detector",
        reconstruction: false,
        operations: &[
            Operation {
                id: "correct",
                title: "Corrections",
                help: "Normalise raw intensity, remove artefacts and retrieve phase",
            },
            Operation {
                id: "correction_preview",
                title: "Correction preview",
                help: "Compare original and corrected detector rows",
            },
            Operation {
                id: "crop",
                title: "Detector crop",
                help: "Select half-open detector row and column ranges",
            },
            Operation {
                id: "bin",
                title: "Detector binning",
                help: "Mean-bin detector pixels",
            },
            Operation {
                id: "select",
                title: "View selection",
                help: "Retain an ordered list of acquisition views",
            },
        ],
    },
    Group {
        title: "Geometry",
        reconstruction: false,
        operations: &[
            Operation {
                id: "poses",
                title: "Pose table",
                help: "Replace or clear the registered six-column pose table",
            },
            Operation {
                id: "align",
                title: "Alignment",
                help: "Fit acquisition geometry and per-view motion",
            },
            Operation {
                id: "region_preview",
                title: "Support estimate",
                help: "Estimate physical object bounds",
            },
            Operation {
                id: "centre_sweep",
                title: "Centre sweep",
                help: "Compare signed centre offsets in detector pixels",
            },
        ],
    },
    Group {
        title: "Reconstruction",
        reconstruction: true,
        operations: &[
            Operation {
                id: "reconstruct",
                title: "Reconstruction",
                help: "Reconstruct the scan using a TomoJAX solver",
            },
            Operation {
                id: "reconstruct_slabs",
                title: "Slab reconstruction",
                help: "Write a reconstruction in bounded-memory slabs",
            },
            Operation {
                id: "plan",
                title: "Reconstruction plan",
                help: "Resolve method, grid, memory estimates and device placement",
            },
            Operation {
                id: "slice",
                title: "Slice preview",
                help: "Reconstruct a registered preview slice",
            },
        ],
    },
    Group {
        title: "Projection",
        reconstruction: true,
        operations: &[
            Operation {
                id: "project",
                title: "Forward projection",
                help: "Project a selected volume using scan geometry",
            },
            Operation {
                id: "backproject",
                title: "Backprojection",
                help: "Apply the matched transpose to projection data",
            },
        ],
    },
];
pub fn lookup(id: &str) -> Option<(&'static Group, &'static Operation)> {
    GROUPS.iter().find_map(|group| {
        group
            .operations
            .iter()
            .find(|op| op.id == id)
            .map(|op| (group, op))
    })
}
pub fn title(id: &str) -> &str {
    lookup(id).map_or_else(
        || match id {
            "export" | "export_array" => "Export",
            "save" => "Save",
            other => other,
        },
        |(_, op)| op.title,
    )
}
pub fn method_name(id: &str) -> String {
    match id {
        "auto" => "Automatic".into(),
        "fourier" => "Fourier".into(),
        "joint" => "Joint alignment".into(),
        "opposite-views" => "Opposite views".into(),
        _ => id.to_uppercase(),
    }
}
