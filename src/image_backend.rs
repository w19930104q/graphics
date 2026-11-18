//! Adapter layer for migrating to a new image backend.
//!
//! `Texture`, `Image` and triangulation historically shared implicit
//! assumptions about the image format:
//!
//! - texture dimensions are integer pixels queried through `ImageSize`
//! - texture coordinates are normalized by dividing source rectangles
//!   in pixels with the texture size
//! - texture operation errors are backend specific types
//!
//! This module makes those assumptions explicit behind the
//! [`BackendTexture`] trait, so a new image backend can be introduced
//! without changing the public drawing API (`Graphics`, `Image`,
//! `TexturePacker`, `triangulation`).
//!
//! Existing textures that implement `ImageSize` satisfy the new trait
//! automatically through a blanket impl, so generic code can migrate
//! one call site at a time. [`TextureAdapter`] is available when a
//! distinct wrapper type is needed, e.g. to give a legacy texture a
//! unified [`ImageError`] error type at the backend boundary.
//!
//! Rendering output is unchanged: the default methods perform exactly
//! the same computations, in the same order, as the legacy code paths.

use crate::{types::SourceRectangle, ImageSize};
use texture::{CreateTexture, Format, TextureOp, TextureSettings, UpdateTexture};

/// Unified error type for texture operations in the new image backend.
///
/// Legacy backends keep their own error types. To migrate, implement
/// `From<LegacyError> for ImageError` for a structural mapping, or use
/// [`ImageError::from_debug`] as a fallback that preserves the `Debug`
/// representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageError {
    /// The texture has zero area or otherwise invalid dimensions.
    InvalidSize,
    /// The pixel data length does not match the texture dimensions.
    SizeMismatch {
        /// The expected number of bytes.
        expected: usize,
        /// The actual number of bytes.
        actual: usize,
    },
    /// A texture update region lies outside the texture bounds.
    OutOfBounds,
    /// The pixel format is not supported by the backend.
    UnsupportedFormat,
    /// A backend specific error without a structural mapping,
    /// preserved through its `Debug` representation.
    Backend(String),
}

impl ImageError {
    /// Maps any legacy texture error into `ImageError`, preserving the
    /// `Debug` representation.
    ///
    /// Backends that want structural error mapping should implement
    /// `From<TheirError> for ImageError` instead.
    pub fn from_debug<E: core::fmt::Debug>(err: E) -> ImageError {
        ImageError::Backend(format!("{:?}", err))
    }
}

impl core::fmt::Display for ImageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ImageError::InvalidSize => write!(f, "invalid texture size"),
            ImageError::SizeMismatch { expected, actual } => write!(
                f,
                "pixel data length {} does not match expected {}",
                actual, expected
            ),
            ImageError::OutOfBounds => write!(f, "texture update region out of bounds"),
            ImageError::UnsupportedFormat => write!(f, "unsupported pixel format"),
            ImageError::Backend(err) => write!(f, "backend error: {}", err),
        }
    }
}

impl std::error::Error for ImageError {}

/// Texture interface assumed by the new image backend.
///
/// This trait is blanket-implemented for all legacy `ImageSize`
/// textures, so existing backend textures work with new generic code
/// without being wrapped. New backends can implement this trait
/// directly to override the format assumptions.
pub trait BackendTexture {
    /// Returns the texture size as `[width, height]` in pixels.
    fn size(&self) -> [u32; 2];

    /// Computes normalized texture coordinates for a source rectangle.
    ///
    /// The default implementation divides the source rectangle in
    /// pixels with the texture size. This matches the legacy
    /// `triangulation::rect_tri_list_uv` computation exactly,
    /// so migrating call sites does not change rendering output.
    fn uv(&self, source_rect: SourceRectangle) -> [[f32; 2]; 6] {
        let [w, h] = self.size();
        let (src_x, src_y, src_w, src_h) = (
            source_rect[0],
            source_rect[1],
            source_rect[2],
            source_rect[3],
        );

        let x1 = src_x as f32 / w as f32;
        let y1 = src_y as f32 / h as f32;
        let x2 = (src_w + src_x) as f32 / w as f32;
        let y2 = (src_h + src_y) as f32 / h as f32;
        [[x1, y1], [x2, y1], [x1, y2], [x2, y1], [x2, y2], [x1, y2]]
    }
}

impl<T: ImageSize> BackendTexture for T {
    fn size(&self) -> [u32; 2] {
        let (w, h) = self.get_size();
        [w, h]
    }
}

/// Adapts a legacy `ImageSize` texture to the new image backend.
///
/// The wrapper is transparent with respect to rendering output: size
/// queries and texture coordinate computations delegate to the wrapped
/// texture. It implements `ImageSize`, so an adapted texture can be
/// used anywhere the legacy texture could, including as
/// `<G as Graphics>::Texture` and in `TexturePacker`.
///
/// Texture creation and update operations delegate to the wrapped
/// texture, but report the unified [`ImageError`] type, mapping legacy
/// errors through `Into<ImageError>`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureAdapter<T: ImageSize>(pub T);

impl<T: ImageSize> TextureAdapter<T> {
    /// Wraps a legacy texture.
    pub fn new(texture: T) -> TextureAdapter<T> {
        TextureAdapter(texture)
    }

    /// Unwraps the legacy texture.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: ImageSize> ImageSize for TextureAdapter<T> {
    fn get_size(&self) -> (u32, u32) {
        self.0.get_size()
    }
}

impl<F, T> TextureOp<F> for TextureAdapter<T>
where
    T: TextureOp<F> + ImageSize,
{
    type Error = ImageError;
}

impl<F, T> CreateTexture<F> for TextureAdapter<T>
where
    T: CreateTexture<F>,
    T::Error: Into<ImageError>,
{
    fn create<S: Into<[u32; 2]>>(
        factory: &mut F,
        format: Format,
        memory: &[u8],
        size: S,
        settings: &TextureSettings,
    ) -> Result<Self, Self::Error> {
        T::create(factory, format, memory, size, settings)
            .map(TextureAdapter)
            .map_err(Into::into)
    }
}

impl<F, T> UpdateTexture<F> for TextureAdapter<T>
where
    T: UpdateTexture<F>,
    T::Error: Into<ImageError>,
{
    fn update<O, S>(
        &mut self,
        factory: &mut F,
        format: Format,
        memory: &[u8],
        offset: O,
        size: S,
    ) -> Result<(), Self::Error>
    where
        O: Into<[u32; 2]>,
        S: Into<[u32; 2]>,
    {
        self.0
            .update(factory, format, memory, offset, size)
            .map_err(Into::into)
    }
}
