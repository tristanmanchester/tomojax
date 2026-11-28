use super::*;
use proptest::prelude::*;
use serde_json::json;
use studio_core::pixels::PlaneSelection;

fn source(file: u8) -> ArrayDescriptor {
    let axes = ["z", "y", "x"].map(|a| {
        json!({"name":a,"unit":"mm","frame":"world",
        "coordinates":{"type":"uniform","origin":0.,"spacing":-2.}})
    });
    serde_json::from_value(
        json!({"storage":{"type":"npy","path":format!("{file}.npy")},
        "shape":[4,4,4],"dtype":"<f4","domain":"test","unit":"1","axes":axes}),
    )
    .unwrap()
}

proptest! {
    #[test]
    fn delayed_decodes_never_cross_a_binding_or_replace_a_newer_plane(
        steps in prop::collection::vec((0u8..6,0u8..2,0usize..4,any::<usize>()),1..100)
    ) {
        let worker = crate::imaging::tests::file_worker();
        let mut images=Imaging::default();

        let mut epoch=0;
        let mut bound=None;
        let mut floor=0;
        let mut wanted=None;
        let mut displayed=None;
        let mut replies:Vec<(u64,Request,bool)>=Vec::new();
        for (step, file, index, choice) in steps {
            match step {
                0..=2 => {
                    let axis=choice%3;
                    images.request_plane(1,source(file),axis,index,32).unwrap();
                    let request=images.slots[&1].request.clone();
                    let binding=(epoch,file,axis);
                    if bound!=Some(binding) {floor=request.generation;displayed=None;}
                    bound=Some(binding);
                    wanted=Some(request.generation);
                    if images.status(1) == &ImagingStatus::Ready { displayed=Some((request.generation,true)); }
                    for exact in [false,true] {replies.push((epoch,request.clone(),exact));}
                }
                3 => {
                    epoch+=1;
                    if file==0 {images.invalidate();} else {images.disconnected();}
                    bound=None;wanted=None;displayed=None;
                    prop_assert!(images.plane(1).is_none());
                }
                _ if !replies.is_empty() => {
                    let (sent_epoch,request,exact)=replies.swap_remove(choice%replies.len());
                    let (axis,i)=request.key.plane.unwrap();
                    let selection=PlaneSelection::new(&request.descriptor.shape,axis,i,32).unwrap();
                    let plane=reader::registered_plane(&worker.files(),&request.descriptor,(axis,i),&selection,vec![i as f32;16],exact,&mut Default::default()).unwrap();
                    let ticket=(request.generation,exact);
                    let expected=sent_epoch==epoch && bound.is_some() && request.generation>=floor
                        && (!exact || wanted==Some(request.generation)) && displayed.is_none_or(|last| ticket>=last);
                    let shown=images.install(Completion{request,asset:Ok(Asset::Plane(Arc::new(plane))),finished:exact,cache_hit:false});
                    prop_assert_eq!(shown,expected);
                    if shown {
                        displayed=Some(ticket);
                        let plane=images.plane(1).unwrap();
                        prop_assert_eq!(plane.index,i);
                        prop_assert_eq!(plane.fixed.last().unwrap().coordinate,-2.*i as f64);
                        prop_assert_eq!(plane.sample(0,0),exact.then_some(i as f32));
                    }
                }
                _ => {}
            }
        }
    }
}

#[test]
fn delayed_crops_cannot_overwrite_a_pan_fit_slice_rebind_or_reconnect() {
    let engine = crate::imaging::tests::file_worker();
    let files = engine.files();
    let completion = |request: &Request, finished: bool, value: f32| {
        let (axis, index) = request.key.plane.unwrap();
        let selection = PlaneSelection::for_region(
            &request.descriptor.shape,
            axis,
            index,
            request.crop.map_or(request.key.edge, |crop| crop.edge),
            request.crop.map(|crop| crop.bounds),
        )
        .unwrap();
        let values = vec![value; selection.count.iter().product()];
        let plane = reader::registered_plane(
            &files,
            &request.descriptor,
            (axis, index),
            &selection,
            values,
            finished,
            &mut Default::default(),
        )
        .unwrap();
        Completion {
            request: request.clone(),
            asset: Ok(Asset::Plane(Arc::new(plane))),
            finished,
            cache_hit: false,
        }
    };
    for change in ["pan", "fit", "slice", "source", "epoch", "connection"] {
        let mut images = Imaging::default();
        let mut descriptor = source(0);
        descriptor.shape = vec![2, 128, 128];
        images
            .request_plane(1, descriptor.clone(), 0, 0, 32)
            .unwrap();
        let base = images.slots[&1].request.clone();
        assert!(images.install(completion(&base, true, 0.)));
        let mut display = super::super::DisplaySettings {
            zoom: 4.,
            ..Default::default()
        };
        assert!(images.view_region(1, Some(&display)));
        let crop = images.slots[&1].request.clone();
        assert!(images.install(completion(&crop, false, 7.)));
        assert_eq!(images.plane(1).unwrap().probe([0.5; 2]), None);
        assert_eq!(images.plane(1).unwrap().probe([0.; 2]), Some(([0, 0], 0.)));
        match change {
            "pan" => {
                display.pan = [2., -1.];
                assert!(images.view_region(1, Some(&display)));
                let latest = images.slots[&1].request.clone();
                assert_eq!(latest.crop.unwrap().bounds, [[80, 112], [0, 16]]);
                assert!(images.install(completion(&latest, true, 2.)));
                assert_eq!(
                    images.plane(1).unwrap().probe([1.5 / 128., 100.5 / 128.]),
                    Some(([100, 1], 2.))
                );
            }
            "fit" => {
                assert!(images.view_region(1, None));
            }
            "slice" => {
                images
                    .request_plane(1, descriptor.clone(), 0, 1, 32)
                    .unwrap();
            }
            "source" => {
                images.request_plane(1, source(1), 0, 0, 32).unwrap();
            }
            "epoch" => {
                images.invalidate();
            }
            "connection" => {
                images.disconnected();
                images
                    .request_plane(1, descriptor.clone(), 0, 1, 32)
                    .unwrap();
                images.request_plane(1, descriptor, 0, 0, 32).unwrap();
                assert!(
                    images.plane(1).is_none(),
                    "old overview survived a reconnect"
                );
            }
            _ => unreachable!(),
        }
        for finished in [false, true] {
            assert!(
                !images.install(completion(&crop, finished, 99.)),
                "{change}"
            );
        }
    }
}
