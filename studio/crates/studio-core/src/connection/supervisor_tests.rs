use super::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn retries_require_time_or_user_action(events in prop::collection::vec((0u8..8, 0u64..40, 0u16..=1000), 1..180)) {
        let mut now = Instant::now();
        let mut retry = Retry::default();
        let mut blocked = false;
        let mut failures = 0u32;
        for (event, seconds, random) in events {
            now += Duration::from_secs(seconds);
            if event == 0 {
                retry.retry_now();
                blocked = false;
            }
            if !retry.begin(now) {
                continue;
            }
            prop_assert!(!blocked);
            if event == 1 {
                retry.connected(now);
                now += Duration::from_secs(seconds);
                if seconds >= 30 {
                    failures = 0;
                }
            }
            let code = match event {
                2 => 3,
                3 => 5,
                4 => 6,
                _ => 4,
            };
            let jitter = f64::from(random) / 1000.0;
            retry.exited(Some(code), now, jitter);
            blocked = matches!(code, 3 | 5 | 6);
            if blocked {
                prop_assert!(!retry.begin(now + Duration::from_secs(1000000)));
            } else {
                let ceiling = (2000u64 << failures.min(8)).min(300000);
                let State::Waiting(at) = retry.state else {
                    unreachable!()
                };
                let delay = (at - now).as_millis();
                prop_assert!((u128::from(ceiling / 2)..=u128::from(ceiling)).contains(&delay));
                if random == 0 {
                    prop_assert_eq!(delay, u128::from(ceiling / 2));
                }
                if random == 1000 {
                    prop_assert_eq!(delay, u128::from(ceiling));
                }
                failures += 1;
            }
        }
    }
}
