//! CPU reference backend used to verify the image backend adapter layer.
//!
//! Renders a fixed scene with a fixed draw state through both the
//! legacy texture path and the `TextureAdapter` path, then compares
//! framebuffer hashes, texture sizes and error mapping.

use std::marker::PhantomData;

use graphics::image::draw_many;
use graphics::image_backend::{BackendTexture, ImageError, TextureAdapter};
use graphics::texture_packer::TexturePacker;
use graphics::types::Matrix2d;
use graphics::{math, DrawState, Graphics, Image, ImageSize};
use texture::{CreateTexture, Format, TextureOp, TextureSettings, UpdateTexture};

// ---------------------------------------------------------------------------
// CPU texture (legacy backend texture)
// ---------------------------------------------------------------------------

/// A CPU RGBA8 texture, standing in for a legacy backend texture.
#[derive(Clone, Debug, PartialEq)]
struct CpuTexture {
    width: u32,
    height: u32,
    /// RGBA8 pixel data, row major.
    data: Vec<u8>,
}

/// Legacy backend specific error type.
#[derive(Clone, Debug, PartialEq, Eq)]
enum CpuTextureError {
    ZeroArea,
    SizeMismatch { expected: usize, actual: usize },
    OutOfBounds,
    UnsupportedFormat,
}

/// Structural error mapping from the legacy error type to the unified
/// `ImageError` of the new image backend.
impl From<CpuTextureError> for ImageError {
    fn from(err: CpuTextureError) -> ImageError {
        match err {
            CpuTextureError::ZeroArea => ImageError::InvalidSize,
            CpuTextureError::SizeMismatch { expected, actual } => {
                ImageError::SizeMismatch { expected, actual }
            }
            CpuTextureError::OutOfBounds => ImageError::OutOfBounds,
            CpuTextureError::UnsupportedFormat => ImageError::UnsupportedFormat,
        }
    }
}

impl ImageSize for CpuTexture {
    fn get_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl TextureOp<()> for CpuTexture {
    type Error = CpuTextureError;
}

impl CreateTexture<()> for CpuTexture {
    fn create<S: Into<[u32; 2]>>(
        _factory: &mut (),
        format: Format,
        memory: &[u8],
        size: S,
        _settings: &TextureSettings,
    ) -> Result<Self, Self::Error> {
        match format {
            Format::Rgba8 => {}
        }
        let size = size.into();
        if size[0] == 0 || size[1] == 0 {
            return Err(CpuTextureError::ZeroArea);
        }
        let expected = 4 * size[0] as usize * size[1] as usize;
        if memory.len() != expected {
            return Err(CpuTextureError::SizeMismatch {
                expected,
                actual: memory.len(),
            });
        }
        Ok(CpuTexture {
            width: size[0],
            height: size[1],
            data: memory.to_vec(),
        })
    }
}

impl UpdateTexture<()> for CpuTexture {
    fn update<O, S>(
        &mut self,
        _factory: &mut (),
        format: Format,
        memory: &[u8],
        offset: O,
        size: S,
    ) -> Result<(), Self::Error>
    where
        O: Into<[u32; 2]>,
        S: Into<[u32; 2]>,
    {
        match format {
            Format::Rgba8 => {}
        }
        let offset = offset.into();
        let size = size.into();
        let expected = 4 * size[0] as usize * size[1] as usize;
        if memory.len() != expected {
            return Err(CpuTextureError::SizeMismatch {
                expected,
                actual: memory.len(),
            });
        }
        if offset[0] + size[0] > self.width || offset[1] + size[1] > self.height {
            return Err(CpuTextureError::OutOfBounds);
        }
        let row_bytes = size[0] as usize * 4;
        for row in 0..size[1] as usize {
            let src = row * row_bytes;
            let dst =
                ((offset[1] as usize + row) * self.width as usize + offset[0] as usize) * 4;
            self.data[dst..dst + row_bytes].copy_from_slice(&memory[src..src + row_bytes]);
        }
        Ok(())
    }
}

/// Makes a deterministic 32x32 test texture.
fn make_texture() -> CpuTexture {
    let (w, h) = (32u32, 32u32);
    let mut data = vec![0u8; 4 * (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = 4 * (y * w + x) as usize;
            data[i] = (x * 8) as u8;
            data[i + 1] = (y * 8) as u8;
            data[i + 2] = ((x + y) * 4) as u8;
            data[i + 3] = 255;
        }
    }
    CpuTexture {
        width: w,
        height: h,
        data,
    }
}

// ---------------------------------------------------------------------------
// CPU graphics backend
// ---------------------------------------------------------------------------

/// Pixel sampling, implemented by both the legacy texture and the adapter.
trait Sample {
    fn sample_nearest(&self, u: f32, v: f32) -> [f32; 4];
}

impl Sample for CpuTexture {
    fn sample_nearest(&self, u: f32, v: f32) -> [f32; 4] {
        let x = (u * self.width as f32)
            .floor()
            .clamp(0.0, self.width as f32 - 1.0) as usize;
        let y = (v * self.height as f32)
            .floor()
            .clamp(0.0, self.height as f32 - 1.0) as usize;
        let i = 4 * (y * self.width as usize + x);
        [
            self.data[i] as f32 / 255.0,
            self.data[i + 1] as f32 / 255.0,
            self.data[i + 2] as f32 / 255.0,
            self.data[i + 3] as f32 / 255.0,
        ]
    }
}

impl Sample for TextureAdapter<CpuTexture> {
    fn sample_nearest(&self, u: f32, v: f32) -> [f32; 4] {
        self.0.sample_nearest(u, v)
    }
}

/// A deterministic CPU graphics backend rendering into a framebuffer.
struct CpuGraphics<T> {
    width: u32,
    height: u32,
    frame: Vec<[f32; 4]>,
    _marker: PhantomData<T>,
}

impl<T> CpuGraphics<T> {
    fn new(width: u32, height: u32) -> Self {
        CpuGraphics {
            width,
            height,
            frame: vec![[0.0; 4]; (width * height) as usize],
            _marker: PhantomData,
        }
    }

    /// FNV-1a hash of the framebuffer.
    fn hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for px in &self.frame {
            for c in px {
                for b in c.to_bits().to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
        }
        h
    }
}

fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
}

/// Fills a triangle, computing the pixel color from barycentric coordinates.
fn fill_triangle(
    frame: &mut [[f32; 4]],
    width: u32,
    height: u32,
    tri: &[[f32; 2]],
    color_at: &mut dyn FnMut(f32, f32, f32) -> [f32; 4],
) {
    let (v0, v1, v2) = (tri[0], tri[1], tri[2]);
    let area = edge(v0, v1, v2);
    if area == 0.0 {
        return;
    }
    let min_x = v0[0].min(v1[0]).min(v2[0]).floor().max(0.0) as u32;
    let min_y = v0[1].min(v1[1]).min(v2[1]).floor().max(0.0) as u32;
    let max_x = (v0[0].max(v1[0]).max(v2[0]).ceil().max(0.0) as u32).min(width);
    let max_y = (v0[1].max(v1[1]).max(v2[1]).ceil().max(0.0) as u32).min(height);
    for y in min_y..max_y {
        for x in min_x..max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let l0 = edge(v1, v2, p) / area;
            let l1 = edge(v2, v0, p) / area;
            let l2 = edge(v0, v1, p) / area;
            if l0 >= 0.0 && l1 >= 0.0 && l2 >= 0.0 {
                frame[(y * width + x) as usize] = color_at(l0, l1, l2);
            }
        }
    }
}

impl<T> Graphics for CpuGraphics<T>
where
    T: ImageSize + Sample,
{
    type Texture = T;

    fn clear_color(&mut self, color: [f32; 4]) {
        for px in &mut self.frame {
            *px = color;
        }
    }

    fn clear_stencil(&mut self, _value: u8) {}

    fn tri_list<F>(&mut self, _draw_state: &DrawState, color: &[f32; 4], mut f: F)
    where
        F: FnMut(&mut dyn FnMut(&[[f32; 2]])),
    {
        let color = *color;
        f(&mut |vertices: &[[f32; 2]]| {
            for tri in vertices.chunks_exact(3) {
                fill_triangle(
                    &mut self.frame,
                    self.width,
                    self.height,
                    tri,
                    &mut |_, _, _| color,
                );
            }
        });
    }

    fn tri_list_c<F>(&mut self, _draw_state: &DrawState, mut f: F)
    where
        F: FnMut(&mut dyn FnMut(&[[f32; 2]], &[[f32; 4]])),
    {
        f(&mut |vertices: &[[f32; 2]], colors: &[[f32; 4]]| {
            for (tri, cols) in vertices.chunks_exact(3).zip(colors.chunks_exact(3)) {
                let (c0, c1, c2) = (cols[0], cols[1], cols[2]);
                fill_triangle(
                    &mut self.frame,
                    self.width,
                    self.height,
                    tri,
                    &mut |l0, l1, l2| {
                        [
                            l0 * c0[0] + l1 * c1[0] + l2 * c2[0],
                            l0 * c0[1] + l1 * c1[1] + l2 * c2[1],
                            l0 * c0[2] + l1 * c1[2] + l2 * c2[2],
                            l0 * c0[3] + l1 * c1[3] + l2 * c2[3],
                        ]
                    },
                );
            }
        });
    }

    fn tri_list_uv<F>(
        &mut self,
        _draw_state: &DrawState,
        color: &[f32; 4],
        texture: &T,
        mut f: F,
    ) where
        F: FnMut(&mut dyn FnMut(&[[f32; 2]], &[[f32; 2]])),
    {
        let color = *color;
        f(&mut |vertices: &[[f32; 2]], uvs: &[[f32; 2]]| {
            for (tri, uv) in vertices.chunks_exact(3).zip(uvs.chunks_exact(3)) {
                let (uv0, uv1, uv2) = (uv[0], uv[1], uv[2]);
                fill_triangle(
                    &mut self.frame,
                    self.width,
                    self.height,
                    tri,
                    &mut |l0, l1, l2| {
                        let u = l0 * uv0[0] + l1 * uv1[0] + l2 * uv2[0];
                        let v = l0 * uv0[1] + l1 * uv1[1] + l2 * uv2[1];
                        let t = texture.sample_nearest(u, v);
                        [
                            t[0] * color[0],
                            t[1] * color[1],
                            t[2] * color[2],
                            t[3] * color[3],
                        ]
                    },
                );
            }
        });
    }

    fn tri_list_uv_c<F>(&mut self, _draw_state: &DrawState, texture: &T, mut f: F)
    where
        F: FnMut(&mut dyn FnMut(&[[f32; 2]], &[[f32; 2]], &[[f32; 4]])),
    {
        f(
            &mut |vertices: &[[f32; 2]], uvs: &[[f32; 2]], colors: &[[f32; 4]]| {
                for ((tri, uv), cols) in vertices
                    .chunks_exact(3)
                    .zip(uvs.chunks_exact(3))
                    .zip(colors.chunks_exact(3))
                {
                    let (uv0, uv1, uv2) = (uv[0], uv[1], uv[2]);
                    let (c0, c1, c2) = (cols[0], cols[1], cols[2]);
                    fill_triangle(
                        &mut self.frame,
                        self.width,
                        self.height,
                        tri,
                        &mut |l0, l1, l2| {
                            let u = l0 * uv0[0] + l1 * uv1[0] + l2 * uv2[0];
                            let v = l0 * uv0[1] + l1 * uv1[1] + l2 * uv2[1];
                            let t = texture.sample_nearest(u, v);
                            [
                                t[0] * (l0 * c0[0] + l1 * c1[0] + l2 * c2[0]),
                                t[1] * (l0 * c0[1] + l1 * c1[1] + l2 * c2[1]),
                                t[2] * (l0 * c0[2] + l1 * c1[2] + l2 * c2[2]),
                                t[3] * (l0 * c0[3] + l1 * c1[3] + l2 * c2[3]),
                            ]
                        },
                    );
                }
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Fixed scene rendered with a fixed draw state
// ---------------------------------------------------------------------------

/// Renders a fixed scene covering the solid and textured code paths.
fn render_scene<G: Graphics>(g: &mut G, texture: &G::Texture) {
    // Fixed draw state for all explicit draws.
    let draw_state = DrawState::default();
    let transform: Matrix2d = math::identity();

    graphics::clear([0.1, 0.2, 0.3, 1.0], g);
    graphics::rectangle([0.8, 0.2, 0.2, 1.0], [8.0, 8.0, 30.0, 20.0], transform, g);
    graphics::ellipse([0.2, 0.8, 0.3, 1.0], [52.0, 10.0, 36.0, 24.0], transform, g);
    graphics::line([0.9, 0.9, 0.1, 1.0], 2.0, [5.0, 80.0, 120.0, 60.0], transform, g);

    // Textured path: goes through `triangulation::rect_tri_list_uv`,
    // which is routed through the `BackendTexture` adapter layer.
    Image::new()
        .rect([20.0, 40.0, 40.0, 40.0])
        .src_rect([4.0, 4.0, 24.0, 24.0])
        .draw(texture, &draw_state, transform, g);
    // Default source rectangle: full texture size.
    Image::new()
        .rect([70.0, 30.0, 32.0, 32.0])
        .draw(texture, &draw_state, transform, g);
    draw_many(
        &[
            ([10.0, 60.0, 16.0, 16.0], [0.0, 0.0, 16.0, 16.0]),
            ([100.0, 60.0, 16.0, 16.0], [16.0, 16.0, 16.0, 16.0]),
        ],
        [1.0, 1.0, 1.0, 1.0],
        texture,
        &draw_state,
        transform,
        g,
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Golden framebuffer hash of the fixed scene on the CPU backend.
/// Guards against output changes introduced by the adapter layer.
const GOLDEN_SCENE_HASH: u64 = 9245818378131271767;

#[test]
fn image_hash_matches_between_legacy_and_adapted_path() {
    let texture = make_texture();

    let mut legacy = CpuGraphics::<CpuTexture>::new(128, 96);
    render_scene(&mut legacy, &texture);

    let mut adapted = CpuGraphics::<TextureAdapter<CpuTexture>>::new(128, 96);
    let adapted_texture = TextureAdapter::new(texture.clone());
    render_scene(&mut adapted, &adapted_texture);

    assert_eq!(
        legacy.hash(),
        adapted.hash(),
        "adapter layer changed rendering output"
    );
    assert_eq!(
        legacy.hash(),
        GOLDEN_SCENE_HASH,
        "rendering output differs from the recorded golden hash"
    );
}

#[test]
fn texture_sizes_match_through_adapter() {
    let texture = make_texture();
    let adapted = TextureAdapter::new(texture.clone());

    // `ImageSize` compatibility is preserved by the adapter.
    assert_eq!(adapted.get_size(), texture.get_size());
    assert_eq!(adapted.get_width(), texture.get_width());
    assert_eq!(adapted.get_height(), texture.get_height());

    // The new backend trait reports identical sizes for both.
    assert_eq!(BackendTexture::size(&texture), [32, 32]);
    assert_eq!(adapted.size(), texture.size());

    // UV computation through the adapter layer matches the legacy
    // triangulation path exactly.
    let src = [4.0, 8.0, 16.0, 24.0];
    assert_eq!(
        BackendTexture::uv(&texture, src),
        graphics::triangulation::rect_tri_list_uv(&texture, src)
    );
    assert_eq!(adapted.uv(src), BackendTexture::uv(&texture, src));

    // `TexturePacker` behaves identically with adapted textures.
    // `create` takes the size of the first tile; the texture is the atlas.
    let mut packer_legacy = TexturePacker::<CpuTexture>::new();
    let mut packer_adapted = TexturePacker::<TextureAdapter<CpuTexture>>::new();
    assert_eq!(
        packer_legacy.create([8, 8], make_texture()),
        packer_adapted.create([8, 8], TextureAdapter::new(make_texture()))
    );
    for size in [[8u32, 8], [16, 8], [8, 16], [7, 5], [32, 32]] {
        let space_legacy = packer_legacy.find_space(size);
        let space_adapted = packer_adapted.find_space(size);
        assert_eq!(space_legacy, space_adapted);
        if let Some(ind) = space_legacy {
            assert_eq!(
                packer_legacy.update(ind, size),
                packer_adapted.update(ind, size)
            );
        }
    }
}

#[test]
fn error_mapping_matches_through_adapter() {
    // Structural mapping table from the legacy error type.
    let cases = vec![
        (CpuTextureError::ZeroArea, ImageError::InvalidSize),
        (
            CpuTextureError::SizeMismatch {
                expected: 64,
                actual: 3,
            },
            ImageError::SizeMismatch {
                expected: 64,
                actual: 3,
            },
        ),
        (CpuTextureError::OutOfBounds, ImageError::OutOfBounds),
        (
            CpuTextureError::UnsupportedFormat,
            ImageError::UnsupportedFormat,
        ),
    ];
    for (legacy, mapped) in cases {
        assert_eq!(ImageError::from(legacy), mapped);
    }

    let settings = TextureSettings::new();

    // Creation with a zero area maps to `ImageError::InvalidSize`.
    let err = <TextureAdapter<CpuTexture> as CreateTexture<()>>::create(
        &mut (),
        Format::Rgba8,
        &[],
        [0u32, 4],
        &settings,
    )
    .unwrap_err();
    assert_eq!(err, ImageError::InvalidSize);

    // Creation with a truncated buffer maps to `ImageError::SizeMismatch`.
    let err = <TextureAdapter<CpuTexture> as CreateTexture<()>>::create(
        &mut (),
        Format::Rgba8,
        &[0u8; 3],
        [4u32, 4],
        &settings,
    )
    .unwrap_err();
    assert_eq!(
        err,
        ImageError::SizeMismatch {
            expected: 64,
            actual: 3
        }
    );

    // An out-of-bounds update maps to `ImageError::OutOfBounds`.
    let mut adapted = TextureAdapter::new(make_texture());
    let err = UpdateTexture::update(
        &mut adapted,
        &mut (),
        Format::Rgba8,
        &[0u8; 16],
        [31u32, 31],
        [2u32, 2],
    )
    .unwrap_err();
    assert_eq!(err, ImageError::OutOfBounds);

    // A successful update through the adapter modifies the inner texture.
    UpdateTexture::update(
        &mut adapted,
        &mut (),
        Format::Rgba8,
        &[7u8; 16],
        [0u32, 0],
        [2u32, 2],
    )
    .unwrap();
    let mut expected = make_texture();
    UpdateTexture::update(&mut expected, &mut (), Format::Rgba8, &[7u8; 16], [0u32, 0], [2u32, 2])
        .unwrap();
    assert_eq!(adapted.into_inner(), expected);

    // Errors without a structural mapping fall back to the `Debug`
    // representation.
    assert_eq!(
        ImageError::from_debug(CpuTextureError::ZeroArea),
        ImageError::Backend("ZeroArea".to_owned())
    );
}
