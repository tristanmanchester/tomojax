struct RayUniform {
    eye: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    extent: vec4<f32>,
    clip_min: vec4<f32>,
    clip_max: vec4<f32>,
    lighting: vec4<f32>,
    dimensions: vec4<f32>,
}
@group(0) @binding(0) var<uniform> u: RayUniform;
@group(0) @binding(1) var volume: texture_3d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var transfer: texture_2d<f32>;

struct Vertex { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) index: u32) -> Vertex {
    let corners = array<vec2<f32>, 3>(vec2(-1.,-1.), vec2(3.,-1.), vec2(-1.,3.));
    let p = corners[index];
    return Vertex(vec4(p,0.,1.), vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5));
}
fn density(p: vec3<f32>) -> f32 {
    return textureSampleLevel(volume, linear_sampler, clamp(p, vec3(0.), vec3(1.)), 0.).r;
}
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), c > vec3(0.04045));
}
@fragment fn fs(input: Vertex) -> @location(0) vec4<f32> {
    let screen = vec2((input.uv.x * 2. - 1.) * u.eye.w, 1. - input.uv.y * 2.);
    let direction = normalize(u.forward.xyz + u.forward.w * (screen.x * u.right.xyz + screen.y * u.up.xyz));
    let lower = (u.clip_min.xyz - 0.5) * u.extent.xyz;
    let upper = (u.clip_max.xyz - 0.5) * u.extent.xyz;
    let inverse = 1. / direction;
    let a = (lower - u.eye.xyz) * inverse;
    let b = (upper - u.eye.xyz) * inverse;
    let near = min(a, b);
    let far = max(a, b);
    let enter = max(max(near.x, near.y), max(near.z, 0.));
    let leave = min(min(far.x, far.y), far.z);
    if leave <= enter || any(upper <= lower) { return vec4(0.); }
    let voxel = u.extent.xyz / u.dimensions.xyz;
    let desired_step = min(min(voxel.x, voxel.y), voxel.z) * u.dimensions.w;
    let step = max(desired_step, (leave - enter) / u.clip_max.w);
    var t = enter;
    var accumulated = vec4(0.);
    let delta = 1. / u.dimensions.xyz;
    let light = normalize(-u.forward.xyz + vec3(-0.3, 0.2, 0.6));
    for (var i = 0u; i < 768u; i += 1u) {
        if t >= leave || accumulated.a > 0.99 { break; }
        let segment = min(step, leave - t);
        let p = (u.eye.xyz + (t + segment * 0.5) * direction) / u.extent.xyz + 0.5;
        let value = density(p);
        if value >= -1.e20 && value <= 1.e20 {
            let scalar = clamp((value - u.right.w) / (u.up.w - u.right.w), 0., 1.);
            let material = textureSampleLevel(transfer, linear_sampler, vec2(scalar, 0.5), 0.);
            let alpha = 1. - exp(-max(material.a, 0.) * u.extent.w * segment * 80.);
            if alpha > 0.0001 {
                var color = srgb_to_linear(material.rgb);
                if u.lighting.w > 0.5 {
                    let gradient = vec3(density(p + vec3(delta.x,0.,0.)) - density(p - vec3(delta.x,0.,0.)),
                        density(p + vec3(0.,delta.y,0.)) - density(p - vec3(0.,delta.y,0.)),
                        density(p + vec3(0.,0.,delta.z)) - density(p - vec3(0.,0.,delta.z))) / (2. * voxel);
                    var normal = -direction;
                    if dot(gradient, gradient) > 1.e-10 { normal = -normalize(gradient); }
                    if dot(normal, -direction) < 0. { normal = -normal; }
                    let lambert = max(dot(normal, light), 0.);
                    let highlight = pow(max(dot(normal, normalize(light - direction)), 0.), 28.);
                    color = color * (u.lighting.x + u.lighting.y * lambert) + vec3(u.lighting.z * highlight);
                }
                accumulated += vec4(color * alpha, alpha) * (1. - accumulated.a);
            }
        }
        t += segment;
    }
    return accumulated;
}
