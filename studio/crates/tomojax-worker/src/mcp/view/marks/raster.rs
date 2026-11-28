// Ported from Pillow (MIT-CMU), src/libImaging/Draw.c at 3c41c095. See THIRD_PARTY_NOTICES.md.
//! The same integer coverage rules for sample footprints and displayed outlines.
use std::ops::RangeInclusive;
type Point = [i64; 2];
pub(super) struct Mask {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}
impl Mask {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; width * height],
        }
    }
    fn span(&mut self, y: i64, a: i64, b: i64, ink: u8) {
        if !(0..self.height as i64).contains(&y) {
            return;
        }
        let (lo, hi) = (a.min(b).max(0), a.max(b).min(self.width as i64 - 1));
        if lo <= hi {
            let row = y as usize * self.width;
            self.pixels[row + lo as usize..=row + hi as usize].fill(ink)
        }
    }
    pub fn rect(&mut self, [x, y]: Point, [u, v]: Point, width: i64, ink: u8) {
        for row in y.max(0)..=v.min(self.height as i64 - 1) {
            if width == 0 || row - y < width || v - row < width {
                self.span(row, x, u, ink)
            } else {
                self.span(row, x, (x + width - 1).min(u), ink);
                self.span(row, (u - width + 1).max(x), u, ink)
            }
        }
    }
    fn line(&mut self, a: Point, b: Point, ink: u8) {
        let delta = [b[0] - a[0], b[1] - a[1]];
        let major = usize::from(delta[1].abs() >= delta[0].abs());
        let minor = 1 - major;
        let n = delta[major].abs();
        if n == 0 {
            self.span(a[1], a[0], a[0], ink);
            return;
        }
        let max = if major == 0 { self.width } else { self.height } as i64;
        for t in a[major].min(b[major]).max(0)..=a[major].max(b[major]).min(max - 1) {
            let k = (t - a[major]).abs();
            let offset = (i128::from(k) * i128::from(delta[minor].abs()) * 2 + i128::from(n))
                / (2 * i128::from(n));
            let mut q = a;
            q[major] = t;
            q[minor] += delta[minor].signum() * offset as i64;
            self.span(q[1], q[0], q[0], ink);
        }
    }
    pub fn stroke(&mut self, a: Point, b: Point, width: i64, ink: u8) {
        if width <= 1 || a == b {
            self.line(a, b, ink);
            return;
        }
        let (dx, dy) = ((b[0] - a[0]) as f64, (b[1] - a[1]) as f64);
        let len = dx.hypot(dy);
        let half = (width - 1) as f64 / 2.;
        let offset = |component: f64, scale: f64| down(component * scale / len);
        let (xmin, xmax) = (offset(dy, down(half) as f64), offset(dy, half.round()));
        let (ymin, ymax) = (offset(dx, down(half) as f64), offset(dx, half.round()));
        self.polygon(
            &[
                [a[0] - xmin, a[1] + ymax],
                [b[0] - xmin, b[1] + ymax],
                [b[0] + xmax, b[1] - ymin],
                [a[0] + xmax, a[1] - ymin],
            ],
            ink,
        );
    }
    pub fn polygon(&mut self, points: &[Point], ink: u8) {
        let mut edges = Vec::new();
        for (a, b) in points.iter().zip(points.iter().cycle().skip(1)) {
            if a[1] == b[1] {
                self.span(a[1], a[0], b[0], ink)
            } else {
                edges.push(Edge::new(*a, *b));
            }
        }
        let Some(top) = points.iter().map(|p| p[1]).min() else {
            return;
        };
        let bottom = points
            .iter()
            .map(|p| p[1])
            .max()
            .unwrap()
            .min(self.height as i64);
        let mut cuts = Vec::new();
        for y in top.max(0)..=bottom {
            cuts.clear();
            for (i, e) in edges
                .iter()
                .enumerate()
                .filter(|(_, e)| e.rows.contains(&y))
            {
                let x = e.at(y);
                if y == *e.rows.end() && y < bottom {
                    cuts.extend([x, x])
                } else {
                    cuts.push(e.corner(y, &edges[..i]))
                }
            }
            cuts.sort_unstable_by(f32::total_cmp);
            for span in cuts.as_chunks::<2>().0 {
                self.span(y, span[0].round() as i64, down(f64::from(span[1])), ink)
            }
        }
    }
    pub fn ellipse(&mut self, [x, y]: Point, [u, v]: Point, width: i64, ink: u8) {
        let (a, b) = (u - x, v - y);
        if a < 0 || b < 0 || u < 0 || v < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return;
        }
        let w = if width == 0 { a + b } else { width };
        if w == 0 {
            return;
        }
        let mut outer = Quarter::new(a, b).peekable();
        let mut inner = Quarter::new(a - 2 * (w - 1), b - 2 * (w - 1)).peekable();
        while let Some([r, row]) = outer.next() {
            while outer.next_if(|p| p[1] == row).is_some() {}
            let mut left = inner.peek().map_or(a % 2, |p| p[0]);
            while let Some(p) = inner.next_if(|p| p[1] <= row) {
                left = p[0];
            }
            for sy in [y + (b - row) / 2, y + (b + row) / 2] {
                self.span(sy, x + (a - r) / 2, x + (a - left) / 2, ink);
                self.span(sy, x + (a + left) / 2, x + (a + r) / 2, ink);
            }
        }
    }
}
fn down(x: f64) -> i64 {
    (x.signum() * (x.abs() - 0.5).ceil()) as i64
}
struct Edge {
    start: Point,
    rows: RangeInclusive<i64>,
    slope: f32,
}
impl Edge {
    fn new(a: Point, b: Point) -> Self {
        Self {
            start: a,
            rows: a[1].min(b[1])..=a[1].max(b[1]),
            slope: (b[0] - a[0]) as f32 / (b[1] - a[1]) as f32,
        }
    }
    fn at(&self, y: i64) -> f32 {
        (y - self.start[1]) as f32 * self.slope + self.start[0] as f32
    }
    fn endpoint(&self, y: i64) -> bool {
        y == *self.rows.start() || y == *self.rows.end()
    }
    fn corner(&self, y: i64, previous: &[Edge]) -> f32 {
        let x = self.at(y);
        if self.slope != 0. && self.endpoint(y) {
            let next = if y == *self.rows.end() { y - 1 } else { y + 1 };
            if let Some(other) = previous
                .iter()
                .find(|e| e.slope != 0. && e.endpoint(y) && e.at(y).round() == x.round())
                && other.rows.contains(&next)
            {
                let (a, b) = (self.at(next), other.at(next));
                if x > a.max(b) + 1. {
                    return a.max(b).round() + 1.;
                }
                if x < a.min(b) - 1. {
                    return a.min(b).round() - 1.;
                }
            }
        }
        x
    }
}
struct Quarter {
    a: i64,
    b: i64,
    next: Option<Point>,
}
impl Quarter {
    fn new(a: i64, b: i64) -> Self {
        Self {
            a,
            b,
            next: (a >= 0 && b >= 0).then_some([a, b % 2]),
        }
    }
}
impl Iterator for Quarter {
    type Item = Point;
    fn next(&mut self) -> Option<Point> {
        let [x, y] = self.next?;
        self.next = if [x, y] == [self.a % 2, self.b] {
            None
        } else {
            let delta = |[u, v]: Point| {
                let (a, b) = (i128::from(self.a), i128::from(self.b));
                (a * a * i128::from(v).pow(2) + b * b * i128::from(u).pow(2) - a * a * b * b).abs()
            };
            [[x, y + 2], [x - 2, y + 2], [x - 2, y]]
                .into_iter()
                .filter(|p| p[0] >= 0)
                .min_by_key(|p| delta(*p))
        };
        Some([x, y])
    }
}
