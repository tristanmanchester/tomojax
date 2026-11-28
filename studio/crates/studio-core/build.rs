fn main() {
    let path = "../../../pyproject.toml";
    println!("cargo:rerun-if-changed={path}");
    let project = std::fs::read_to_string(path).expect("read pyproject.toml");
    let version = project
        .lines()
        .find_map(|line| line.strip_prefix("version = \"")?.strip_suffix('"'))
        .expect("TomoJAX package version");
    println!("cargo:rustc-env=TOMOJAX_VERSION={version}");
}
