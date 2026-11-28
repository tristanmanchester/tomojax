//! Lend the foreground terminal to an interactive ssh process group. OpenSSH
//! reads /dev/tty even though its protocol stdin is closed.
use std::{fs::File, io, mem::MaybeUninit, os::fd::AsRawFd, process::Child};

pub(super) struct Terminal {
    tty: File,
    previous: libc::pid_t,
    settings: libc::termios,
}
impl Terminal {
    pub fn lend(child: &Child) -> io::Result<Option<Self>> {
        let tty = match File::options().read(true).write(true).open("/dev/tty") {
            Ok(tty) => tty,
            Err(e) if e.raw_os_error() == Some(libc::ENXIO) => return Ok(None),
            Err(e) => return Err(e),
        };
        let fd = tty.as_raw_fd();
        // SAFETY: fd stays open; tcgetattr initializes settings on success.
        unsafe {
            let previous = libc::tcgetpgrp(fd);
            if previous != libc::getpgrp() {
                return Ok(None);
            }
            let mut settings = MaybeUninit::uninit();
            if libc::tcgetattr(fd, settings.as_mut_ptr()) < 0
                || libc::tcsetpgrp(fd, child.id() as i32) < 0
            {
                return Err(io::Error::last_os_error());
            }
            // ssh may already have tried to read while its group was background.
            libc::kill(child.id() as i32, libc::SIGCONT);
            Ok(Some(Self {
                tty,
                previous,
                settings: settings.assume_init(),
            }))
        }
    }
}
impl Drop for Terminal {
    fn drop(&mut self) {
        let fd = self.tty.as_raw_fd();
        // SAFETY: restore our original terminal/group. POSIX permits this call
        // from a background group when SIGTTOU is blocked on the calling thread.
        unsafe {
            let mut blocked = std::mem::zeroed();
            libc::sigemptyset(&mut blocked);
            libc::sigaddset(&mut blocked, libc::SIGTTOU);
            let mut original = MaybeUninit::uninit();
            if libc::pthread_sigmask(libc::SIG_BLOCK, &blocked, original.as_mut_ptr()) == 0 {
                if libc::tcsetpgrp(fd, self.previous) == 0 {
                    libc::tcsetattr(fd, libc::TCSANOW, &self.settings);
                }
                libc::pthread_sigmask(libc::SIG_SETMASK, original.as_ptr(), std::ptr::null_mut());
            }
        }
    }
}
