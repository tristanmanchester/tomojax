use super::*;
use crate::protocol::{Response, ResultValue};
use proptest::prelude::*;
use std::io::{Cursor, Write};

const RAMP: &[u8] = include_bytes!("../../tests/fixtures/ramp.f32.zst");
const GREY: &[u8] = include_bytes!("../../tests/fixtures/ramp.jpg");

fn block(wire: &mut Vec<u8>, id: &str, bytes: &[u8]) {
    writeln!(
        wire,
        "{}",
        json!({"type":"block","request":id,"bytes":bytes.len()})
    )
    .unwrap();
    wire.extend_from_slice(bytes);
}

fn terminal(wire: &mut Vec<u8>, id: &str, encoding: Encoding, shape: [usize; 2], size: usize) {
    writeln!(
        wire,
        "{}",
        json!({"type":"result","request":id,"result":{
        "type":"pixels","shape":shape,"encoding":encoding,"range":[0,255],"bytes":size}})
    )
    .unwrap();
}

#[test]
fn interleaved_blocks_decode_handmade_ramps_and_return_even_discarded_credit() {
    let mut data = Vec::new();
    block(&mut data, "old", b"\xff\0\n");
    block(&mut data, "a", &RAMP[..7]);
    block(&mut data, "b", GREY);
    terminal(&mut data, "b", Encoding::Jpeg, [8, 32], GREY.len());
    writeln!(data,"{}",json!({"type":"error","request":"old","error":{"schema":2,"event":"error","kind":"superseded","message":"superseded"}})).unwrap();
    block(&mut data, "a", &RAMP[7..]);
    terminal(&mut data, "a", Encoding::Float32, [2, 4], RAMP.len());
    let mut credit = 0;
    let mut wire = wire::WireReader::new(Cursor::new(data));
    let mut read = || {
        wire.next(|n| {
            if let wire::Flow::Credit(bytes) = n {
                credit += bytes;
            }
            Ok(())
        })
        .unwrap()
        .unwrap()
    };
    let Response::Result {
        result: ResultValue::Pixels(grey),
        ..
    } = read()
    else {
        panic!()
    };
    let slab = PlaneSelection::new(&[8, 32], 0, 0, 32).unwrap();
    let samples = grey.decode(&slab, Encoding::Jpeg).unwrap();
    for row in samples.as_chunks::<32>().0 {
        for (x, value) in row.iter().enumerate() {
            assert_eq!(*value, [0., 63., 128., 255.][x / 8]);
        }
    }
    assert!(matches!(read(), Response::Error { .. }));
    let Response::Result {
        result: ResultValue::Pixels(ramp),
        ..
    } = read()
    else {
        panic!()
    };
    assert_eq!(
        ramp.decode(
            &PlaneSelection::new(&[2, 4], 0, 0, 64).unwrap(),
            Encoding::Float32
        )
        .unwrap(),
        [-3.5, -1.25, 0., 0.125, 0.5, 1., 2., 4.]
    );
    assert_eq!(credit, RAMP.len() + GREY.len() + 3);
}

#[test]
fn stalled_decoder_keeps_the_latest_frame_and_all_reliable_messages() {
    let queue = FrameQueue::new(3);
    let mut bytes = Vec::new();
    for i in 0..100 {
        let id = i.to_string();
        block(&mut bytes, &id, GREY);
        terminal(&mut bytes, &id, Encoding::Jpeg, [8, 32], GREY.len());
    }
    let mut reader = wire::WireReader::new(Cursor::new(bytes));
    let mut returned = 0;
    queue.push(None, "state".to_string());
    while let Some(response) = reader
        .next(|n| {
            if let wire::Flow::Credit(bytes) = n {
                returned += bytes;
            }
            Ok(())
        })
        .unwrap()
    {
        queue.push(Some(1), response.request().unwrap().to_owned());
    }
    assert_eq!(queue.pop().as_deref(), Some("state"));
    assert_eq!(queue.pop().as_deref(), Some("99"));
    queue.push(None, "state".into());
    queue.push(Some(1), "old moving frame".into());
    queue.push(Some(2), "another pane".into());
    queue.push(None, "settled".into()); // evicts the older moving frame at capacity
    assert_eq!(queue.pop().as_deref(), Some("state"));
    assert_eq!(queue.pop().as_deref(), Some("another pane"));
    assert_eq!(queue.pop().as_deref(), Some("settled"));
    assert_eq!(returned, 100 * GREY.len());
}

#[test]
fn motion_coalesces_while_jpeg_is_in_flight_and_interrupts_lossless_immediately() {
    let mut pane = PaneStream::Idle;
    assert_eq!(pane.next(1), Some(Encoding::Jpeg));
    pane = PaneStream::Waiting(1, Encoding::Jpeg);
    assert_eq!(pane.next(2), None);
    assert!(pane.receive(1, Encoding::Jpeg));
    assert_eq!(pane.next(3), Some(Encoding::Jpeg));
    pane = PaneStream::Waiting(3, Encoding::Jpeg);
    assert!(pane.receive(3, Encoding::Jpeg));
    assert_eq!(pane.next(3), Some(Encoding::Float32));
    pane = PaneStream::Waiting(3, Encoding::Float32);
    assert_eq!(pane.next(4), Some(Encoding::Jpeg));
    assert!(pane.receive(3, Encoding::Float32));
    assert_eq!(pane.next(3), None);
}

proptest! {
    #[test]
    fn only_the_current_wire_request_can_deliver(actions in prop::collection::vec((0u8..4,any::<usize>()), 1..200)) {
        let mut pane = PaneStream::Idle;
        let mut wanted = 0;
        let mut pending: Vec<(u64, Encoding)> = Vec::new();
        let mut latest = None;
        for (action,n) in actions {
            match action {
                0 => wanted+=1,
                1 => { wanted+=1; latest=None; pane=PaneStream::Idle; } // clear, invalidate or reconnect
                _ if !pending.is_empty() => {
                    let i = n % pending.len();
                    let ticket = pending.swap_remove(i);
                    let accepted = pane.receive(ticket.0,ticket.1);
                    prop_assert_eq!(accepted, latest == Some(ticket));
                    if accepted {latest=None;}
                }
                _ => {}
            }
            if let Some(encoding) = pane.next(wanted) {
                pane = PaneStream::Waiting(wanted,encoding);
                pending.push((wanted,encoding));
                latest=Some((wanted,encoding));
            }
        }
    }
}

#[test]
fn pco_frames_preserve_scalar_bits_and_reject_a_wrong_declared_count() {
    let bits = [0, 0x80000000, 0x7fc12345, 0xff800000, 1, 0x7f800000];
    let values = bits.map(f32::from_bits);
    let packed = pco::standalone::simple_compress(&values, &pco::ChunkConfig::default()).unwrap();
    for n in [bits.len(), bits.len() - 1, bits.len() + 1] {
        let mut data = vec![];
        block(&mut data, "p", &packed);
        terminal(&mut data, "p", Encoding::Pco32, [1, n], packed.len());
        let mut wire = wire::WireReader::new(Cursor::new(data));
        let Some(Response::Result {
            result: ResultValue::Pixels(frame),
            ..
        }) = wire.next(|_| Ok(())).unwrap()
        else {
            panic!("missing pixel terminal")
        };
        let result = frame.floats(&[1, n]);
        if n == bits.len() {
            assert_eq!(
                result
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                bits
            );
        } else {
            assert!(result.is_err());
        }
    }
}
