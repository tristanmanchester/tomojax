//! Borrow the backing file of an open POSIX HDF5 handle, including external links.
use std::fs::File;
use std::os::fd::BorrowedFd;

pub(super) fn descriptor(file: &hdf5::File) -> Option<File> {
    if !matches!(
        file.access_plist().ok()?.get_driver().ok()?,
        hdf5::plist::file_access::FileDriver::Sec2
    ) {
        return None;
    }
    hdf5::sync::sync(|| {
        let mut handle = std::ptr::null_mut();
        // The sec2 driver returns a pointer to its open POSIX file descriptor.
        // The HDF5 File above outlives the borrow and we own the duplicated fd.
        unsafe {
            if hdf5_sys::h5f::H5Fget_vfd_handle(file.id(), hdf5_sys::h5p::H5P_DEFAULT, &mut handle)
                < 0
                || handle.is_null()
            {
                return None;
            }
            let fd = BorrowedFd::borrow_raw(*handle.cast::<std::os::fd::RawFd>());
            Some(File::from(fd.try_clone_to_owned().ok()?))
        }
    })
}
