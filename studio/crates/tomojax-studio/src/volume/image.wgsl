struct ImageUniform {
    rect: vec4<f32>,
    canvas: vec4<f32>,
    viewport: vec4<f32>,
    window: vec4<f32>,
    info: vec4<f32>,
    clip: vec4<f32>,
    fade: vec4<u32>,
}
@group(0) @binding(0) var<uniform> u: ImageUniform;
@group(0) @binding(1) var image: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corners = array<vec2<f32>, 6>(vec2(0.,0.), vec2(1.,0.), vec2(0.,1.), vec2(0.,1.), vec2(1.,0.), vec2(1.,1.));
    let point = u.canvas.xy + corners[index] * u.canvas.zw - u.viewport.zw;
    return vec4(point.x / u.viewport.x * 2. - 1., 1. - point.y / u.viewport.y * 2., 0., 1.);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), c > vec3(0.04045));
}
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return select(c * 12.92, 1.055 * pow(max(c, vec3(0.)), vec3(1. / 2.4)) - 0.055, c > vec3(0.0031308));
}
fn palette(t: f32, kind: u32) -> vec3<f32> {
    if kind == 0u { return vec3(t); }
    var stops = array<vec3<f32>, 5>(vec3(0.267,0.005,0.329), vec3(0.231,0.322,0.545), vec3(0.129,0.569,0.549), vec3(0.369,0.788,0.384), vec3(0.993,0.906,0.144));
    if kind == 2u {
        stops = array<vec3<f32>, 5>(vec3(0.001,0.,0.016), vec3(0.318,0.071,0.486), vec3(0.718,0.216,0.475), vec3(0.988,0.537,0.380), vec3(0.988,0.992,0.749));
    }
    if kind == 3u {
        stops = array<vec3<f32>, 5>(vec3(0.06,0.10,0.10), vec3(0.22,0.34,0.30), vec3(0.49,0.58,0.46), vec3(0.78,0.76,0.60), vec3(0.99,0.97,0.87));
    }
    let at = clamp(t, 0., 1.) * 4.;
    let low = min(u32(at), 3u);
    return mix(stops[low], stops[low + 1u], at - f32(low));
}
fn fetch(uv: vec2<f32>) -> vec4<f32> {
    let size = textureDimensions(image);
    let p = clamp(uv * vec2<f32>(size) - 0.5, vec2(0.), vec2<f32>(size) - 1.);
    let low = vec2<i32>(floor(p));
    let high = min(low + 1, vec2<i32>(size) - 1);
    let f = fract(p);
    return mix(mix(textureLoad(image, low, 0), textureLoad(image, vec2(high.x,low.y), 0), f.x),
        mix(textureLoad(image, vec2(low.x,high.y), 0), textureLoad(image, high, 0), f.x), f.y);
}
@fragment fn fs(@builtin(position) point: vec4<f32>) -> @location(0) vec4<f32> {
    let absolute = point.xy + u.viewport.zw;
    let uv = (absolute - u.rect.xy) / u.rect.zw;
    if any(uv < vec2(0.)) || any(uv > vec2(1.)) { discard; }
    let mask = clip_alpha(absolute, u.clip, u.fade);
    let sample = fetch(uv);
    if u.info.y > 0.5 {
        let unpremultiplied = sample.rgb / max(sample.a, 0.00001);
        let rgb = select(unpremultiplied, linear_to_srgb(unpremultiplied), u.info.x > 0.5);
        return vec4(rgb * sample.a, sample.a) * mask;
    }
    if !(sample.r >= -3.4e38 && sample.r <= 3.4e38) {
        let checker = f32((u32(point.x / 5.) + u32(point.y / 5.)) % 2u);
        return vec4(vec3(0.035 + checker * 0.025), 1.) * mask;
    }
    var t = clamp((sample.r - u.window.x) / (u.window.y - u.window.x), 0., 1.);
    if u.window.w > 0.5 { t = 1. - t; }
    let rgb = palette(t, u32(u.window.z));
    return vec4(select(srgb_to_linear(rgb), rgb, u.info.x > 0.5), 1.) * mask;
}
