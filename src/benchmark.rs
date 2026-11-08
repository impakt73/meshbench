use crate::cache::CacheConfig;
use crate::cluster::{ClusterConfig, PrimitiveClusters, build_clusters, prepare_primitive};
use crate::geometry::PreparedPrimitive;
use crate::gltf_input::load_gltf;
use crate::report::{BenchmarkReport, PathReport, PrimitiveReport, StorageMetrics};
use crate::simulator::{simulate_localized, simulate_standard};
use anyhow::{Result, bail};
use std::mem::size_of;
use std::path::Path;

pub struct BenchmarkOptions {
    pub cluster_config: ClusterConfig,
    pub cache_config: CacheConfig,
}

pub fn run(input: &Path, options: &BenchmarkOptions) -> Result<BenchmarkReport> {
    options.cluster_config.validate()?;
    options.cache_config.validate()?;

    let loaded = load_gltf(input)?;
    let mut warnings = loaded.warnings;

    let mut primitives = Vec::new();
    let mut clusters = Vec::new();
    let mut standard_storage = Vec::new();
    let mut localized_storage = Vec::new();
    let mut primitive_reports = Vec::new();

    for packed in &loaded.primitives {
        let prepared = prepare_primitive(packed);
        if prepared.indices.is_empty() {
            warnings.push(format!("skipping {}: no triangles", prepared.name));
            continue;
        }

        let cluster = build_clusters(&prepared, options.cluster_config)?;
        if cluster.standard.is_empty() {
            warnings.push(format!("skipping {}: no clusters", prepared.name));
            continue;
        }

        let standard = storage_standard(&prepared, &cluster);
        let localized = storage_localized(&prepared, &cluster);
        primitive_reports.push(PrimitiveReport {
            name: prepared.name.clone(),
            vertex_count: prepared.vertex_count as u64,
            triangle_count: prepared.triangle_count() as u64,
            vertex_stride: prepared.stride,
            cluster_count: cluster.standard.len() as u64,
            attributes: prepared.attributes.clone(),
            standard_storage: standard.clone(),
            localized_storage: localized.clone(),
        });
        standard_storage.push(standard);
        localized_storage.push(localized);
        primitives.push(prepared);
        clusters.push(cluster);
    }

    if primitives.is_empty() {
        bail!(
            "{} did not contain any renderable triangle geometry",
            input.display()
        );
    }

    let mut standard_total = StorageMetrics::default();
    let mut localized_total = StorageMetrics::default();
    for storage in &standard_storage {
        standard_total.add(storage);
    }
    for storage in &localized_storage {
        localized_total.add(storage);
    }
    standard_total.finalize();
    localized_total.finalize();

    let total_vertices = primitives
        .iter()
        .map(|primitive| primitive.vertex_count as u64)
        .sum();
    let total_triangles = primitives
        .iter()
        .map(|primitive| primitive.triangle_count() as u64)
        .sum();
    let total_clusters = clusters
        .iter()
        .map(|cluster| cluster.standard.len() as u64)
        .sum();

    let mut standard_memory = simulate_standard(&primitives, &clusters, options.cache_config);
    let mut localized_memory = simulate_localized(&primitives, &clusters, options.cache_config);
    standard_memory.set_workload(total_triangles, total_clusters);
    localized_memory.set_workload(total_triangles, total_clusters);

    Ok(BenchmarkReport {
        input: input.display().to_string(),
        cluster_config: options.cluster_config,
        cache_config: options.cache_config,
        primitive_count: primitives.len(),
        total_vertices,
        total_triangles,
        total_clusters,
        standard: PathReport {
            path: "standard".to_string(),
            storage: standard_total,
            memory: standard_memory,
        },
        localized: PathReport {
            path: "localized".to_string(),
            storage: localized_total,
            memory: localized_memory,
        },
        primitives: primitive_reports,
        warnings,
    })
}

fn storage_standard(primitive: &PreparedPrimitive, cluster: &PrimitiveClusters) -> StorageMetrics {
    let descriptor_size = size_of::<meshopt::ffi::meshopt_Meshlet>() as u64;
    let cluster_count = cluster.standard.len() as u64;
    let mut storage = StorageMetrics {
        cluster_count,
        vertex_count: primitive.vertex_count as u64,
        triangle_count: primitive.triangle_count() as u64,
        descriptor_bytes: cluster_count * descriptor_size,
        vertex_indirection_bytes: cluster.standard.vertices.len() as u64 * size_of::<u32>() as u64,
        vertex_data_bytes: primitive.vertex_count as u64 * primitive.stride as u64,
        triangle_index_bytes: cluster.standard.triangles.len() as u64,
        ..StorageMetrics::default()
    };
    storage.finalize();
    storage
}

fn storage_localized(primitive: &PreparedPrimitive, cluster: &PrimitiveClusters) -> StorageMetrics {
    let descriptor_size = size_of::<meshopt::ffi::meshopt_Meshlet>() as u64;
    let cluster_count = cluster.localized.len() as u64;
    let referenced_vertex_bytes = cluster.referenced_vertex_count as u64 * primitive.stride as u64;
    let mut storage = StorageMetrics {
        cluster_count,
        vertex_count: primitive.vertex_count as u64,
        triangle_count: primitive.triangle_count() as u64,
        descriptor_bytes: cluster_count * descriptor_size,
        vertex_indirection_bytes: 0,
        vertex_data_bytes: cluster.localized.vertices.len() as u64,
        triangle_index_bytes: cluster.localized.triangles.len() as u64,
        duplicate_vertex_bytes: (cluster.localized.vertices.len() as u64)
            .saturating_sub(referenced_vertex_bytes),
        ..StorageMetrics::default()
    };
    storage.finalize();
    storage
}
