//! The plain-language formatters for positions and file sizes.
use super::{coordinate, files::file_size};

#[test]
fn slice_positions_drop_padding_and_name_units() {
    assert_eq!(
        coordinate("view", 22.5, "degree").as_deref(),
        Some("view = 22.5°")
    );
    assert_eq!(
        coordinate("z", 0.5, "scan_length").as_deref(),
        Some("z = 0.5 scan unit")
    );
    assert_eq!(coordinate("y", -3., "1").as_deref(), Some("y = -3"));
    // The slice counter already shows an index position.
    assert_eq!(coordinate("view", 4., "index"), None);
}

#[test]
fn file_sizes_read_in_decimal_units() {
    assert_eq!(file_size(1), "1 byte");
    assert_eq!(file_size(8), "8 bytes");
    assert_eq!(file_size(395_264), "395 KB");
    assert_eq!(file_size(1_250_000), "1.2 MB");
    assert_eq!(file_size(100 * 1024 * 1024), "105 MB");
    assert_eq!(file_size(7_600_000_000_000), "7.6 TB");
}
