// Ported from Pillow src/libImaging/{Resample,Reduce}.c and src/PIL/Image.py at 3c41c095.
use image::{ImageBuffer, Pixel};
use std::f64::consts::PI;

pub enum Filter {
    Lanczos,
    Bicubic,
}

impl Filter {
    fn weight(&self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Self::Lanczos if x == 0. => 1.,
            Self::Lanczos if x < 3. => {
                let a = PI * x;
                let b = PI * (x / 3.);
                (a.sin() / a) * (b.sin() / b)
            }
            Self::Bicubic if x < 1. => (1.5 * x - 2.5) * x * x + 1.,
            Self::Bicubic if x < 2. => -0.5 * (((x - 5.) * x + 8.) * x - 4.),
            _ => 0.,
        }
    }

    fn weights(&self, input: u32, extent: f64, output: u32) -> Vec<(u32, Vec<i32>)> {
        let scale = extent / f64::from(output);
        let stretch = scale.max(1.);
        let radius = stretch
            * match self {
                Self::Lanczos => 3.,
                Self::Bicubic => 2.,
            };
        (0..output)
            .map(|i| {
                let center = (f64::from(i) + 0.5) * scale;
                let first = (center - radius + 0.5) as u32;
                let end = ((center + radius + 0.5) as u32).min(input);
                let weights: Vec<_> = (first..end)
                    .map(|j| self.weight((f64::from(j) - center + 0.5) * stretch.recip()))
                    .collect();
                let sum: f64 = weights.iter().sum();
                (
                    first,
                    weights
                        .into_iter()
                        .map(|w| ((w / sum) * f64::from(1 << 22)).round() as i32)
                        .collect(),
                )
            })
            .collect()
    }
}

type Buffer<P> = ImageBuffer<P, Vec<u8>>;

fn filter_axis<P: Pixel<Subpixel = u8>>(
    image: Buffer<P>,
    extent: f64,
    output: u32,
    vertical: bool,
    filter: &Filter,
) -> Buffer<P> {
    let (width, height) = image.dimensions();
    let input = if vertical { height } else { width };
    if input == output && f64::from(input) == extent {
        return image;
    }
    let weights = filter.weights(input, extent, output);
    let (width, height) = if vertical {
        (width, output)
    } else {
        (output, height)
    };
    ImageBuffer::from_fn(width, height, |x, y| {
        let (first, coeffs) = &weights[if vertical { y } else { x } as usize];
        let mut pixel = *image.get_pixel(
            if vertical { x } else { *first },
            if vertical { *first } else { y },
        );
        for (channel, value) in pixel.channels_mut().iter_mut().enumerate() {
            let mut sum = 1 << 21;
            for (i, coeff) in coeffs.iter().enumerate() {
                let j = first + i as u32;
                let sample = if vertical {
                    image.get_pixel(x, j)
                } else {
                    image.get_pixel(j, y)
                };
                sum += coeff * i32::from(sample.channels()[channel]);
            }
            *value = (sum >> 22).clamp(0, 255) as u8;
        }
        pixel
    })
}

fn reduce<P: Pixel<Subpixel = u8>>(image: Buffer<P>, fx: u32, fy: u32) -> Buffer<P> {
    if (fx, fy) == (1, 1) {
        return image;
    }
    ImageBuffer::from_fn(
        image.width().div_ceil(fx),
        image.height().div_ceil(fy),
        |x, y| {
            let (x0, y0) = (x * fx, y * fy);
            let w = fx.min(image.width() - x0);
            let h = fy.min(image.height() - y0);
            let n = u64::from(w) * u64::from(h);
            let multiplier = ((1u32 << 24) as f32 / n as f32) as u64;
            let mut pixel = *image.get_pixel(x0, y0);
            for (k, v) in pixel.channels_mut().iter_mut().enumerate() {
                let mut sum = n / 2;
                for j in y0..y0 + h {
                    for i in x0..x0 + w {
                        sum += u64::from(image.get_pixel(i, j).channels()[k]);
                    }
                }
                *v = ((sum * multiplier) >> 24) as u8;
            }
            pixel
        },
    )
}

pub fn fit<P: Pixel<Subpixel = u8>>(image: Buffer<P>, filter: Filter) -> Buffer<P> {
    const EDGE: u32 = 1024;
    let (w, h) = image.dimensions();
    if w.max(h) <= EDGE {
        return image;
    }
    let aspect = f64::from(w) / f64::from(h);
    let side = f64::from(EDGE);
    let nearest = |ideal: f64, err: &dyn Fn(f64) -> f64| {
        let (a, b) = (ideal.floor(), ideal.ceil());
        (if err(a) <= err(b) { a } else { b } as u32).max(1)
    };
    let (out_w, out_h) = if w <= h {
        (nearest(side * aspect, &|n| (aspect - n / side).abs()), EDGE)
    } else {
        (
            EDGE,
            nearest(side / aspect, &|n| {
                if n == 0. {
                    0.
                } else {
                    (aspect - side / n).abs()
                }
            }),
        )
    };
    let (fx, fy) = ((w / out_w / 2).max(1), (h / out_h / 2).max(1));
    let image = reduce(image, fx, fy);
    // Pillow passes the reduced source box through single-precision coordinates.
    let ex = f64::from((f64::from(w) / f64::from(fx)) as f32);
    let ey = f64::from((f64::from(h) / f64::from(fy)) as f32);
    if u64::from(image.height()) > 100 * u64::from(image.width()) {
        let vertical = filter_axis(image, ey, out_h, true, &filter);
        filter_axis(vertical, ex, out_w, false, &filter)
    } else {
        let horizontal = filter_axis(image, ex, out_w, false, &filter);
        filter_axis(horizontal, ey, out_h, true, &filter)
    }
}
