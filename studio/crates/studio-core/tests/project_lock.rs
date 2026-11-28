//! A worker can be between fork and exec when the last project window closes.
use std::{
    io::{self, Read, Write},
    os::{
        fd::AsRawFd,
        unix::{net::UnixStream, process::CommandExt},
    },
    process::Command,
    thread,
    time::Duration,
};
use studio_core::project::Project;

#[test]
fn project_reopens_while_a_child_still_has_the_writers_file_description() {
    let directory = tempfile::tempdir().unwrap();
    let project = Project::open(directory.path(), true).unwrap();
    let (mut parent, child) = UnixStream::pair().unwrap();
    let timeout = Some(Duration::from_secs(10));
    parent.set_read_timeout(timeout).unwrap();
    child.set_read_timeout(timeout).unwrap();
    let fork = thread::spawn(move || {
        let mut command = Command::new("/bin/true");
        // SAFETY: between fork and exec the hook uses only read and write on a
        // live Unix socket. The parent releases the child after trying the reopen.
        unsafe {
            command.pre_exec(move || {
                let mut byte = 1_u8;
                let fd = child.as_raw_fd();
                if libc::write(fd, (&raw const byte).cast(), 1) != 1
                    || libc::read(fd, (&raw mut byte).cast(), 1) != 1
                {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        command.status().unwrap()
    });
    parent.read_exact(&mut [0]).unwrap();
    drop(project);
    let reopened = Project::open(directory.path(), false);
    parent.write_all(&[1]).unwrap();
    assert!(fork.join().unwrap().success());
    reopened.unwrap();
}
