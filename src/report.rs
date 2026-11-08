use crate::cache::CacheConfig;
use crate::cluster::ClusterConfig;
use crate::geometry::VertexAttribute;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadCategory {
    Descriptor,
    VertexIndirection,
    PositionData,
    AttributeData,
    TriangleIndex,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LoadCategoryMetrics {
    pub logical_bytes: u64,
    pub sector_requests: u64,
    pub dram_sectors: u64,
    pub dram_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LoadBreakdown {
    pub descriptors: LoadCategoryMetrics,
    pub vertex_indirection: LoadCategoryMetrics,
    pub position_data: LoadCategoryMetrics,
    pub attribute_data: LoadCategoryMetrics,
    pub triangle_indices: LoadCategoryMetrics,
}

impl LoadBreakdown {
    pub fn get_mut(&mut self, category: LoadCategory) -> &mut LoadCategoryMetrics {
        match category {
            LoadCategory::Descriptor => &mut self.descriptors,
            LoadCategory::VertexIndirection => &mut self.vertex_indirection,
            LoadCategory::PositionData => &mut self.position_data,
            LoadCategory::AttributeData => &mut self.attribute_data,
            LoadCategory::TriangleIndex => &mut self.triangle_indices,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MemoryMetrics {
    pub logical_bytes: u64,
    pub l1_sector_requests: u64,
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l1_hit_rate: f64,
    pub l2_sector_requests: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l2_hit_rate: f64,
    pub unique_lines: u64,
    pub unique_sectors: u64,
    pub dram_sectors: u64,
    pub dram_lines: u64,
    pub dram_bytes: u64,
    pub line_amplification: f64,
    pub memory_requests: u64,
    pub waves: u64,
    pub wave_groups: u64,
    pub estimated_latency_cycles: u64,
    pub bandwidth_latency_cycles: u64,
    pub dram_bytes_per_cycle: f64,
    pub logical_bytes_per_triangle: f64,
    pub dram_bytes_per_triangle: f64,
    pub logical_bytes_per_cluster: f64,
    pub dram_bytes_per_cluster: f64,
    pub breakdown: LoadBreakdown,
}

impl MemoryMetrics {
    pub fn finalize(&mut self) {
        self.l1_misses = self.l1_sector_requests.saturating_sub(self.l1_hits);
        self.l2_misses = self.l2_sector_requests.saturating_sub(self.l2_hits);
        self.l1_hit_rate = ratio(self.l1_hits, self.l1_sector_requests);
        self.l2_hit_rate = ratio(self.l2_hits, self.l2_sector_requests);
        self.line_amplification = ratio(self.dram_bytes, self.logical_bytes);
        self.dram_bytes_per_cycle = ratio(self.dram_bytes, self.estimated_latency_cycles);
    }

    pub fn set_workload(&mut self, triangles: u64, clusters: u64) {
        self.logical_bytes_per_triangle = ratio(self.logical_bytes, triangles);
        self.dram_bytes_per_triangle = ratio(self.dram_bytes, triangles);
        self.logical_bytes_per_cluster = ratio(self.logical_bytes, clusters);
        self.dram_bytes_per_cluster = ratio(self.dram_bytes, clusters);
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct StorageMetrics {
    pub cluster_count: u64,
    pub vertex_count: u64,
    pub triangle_count: u64,
    pub descriptor_bytes: u64,
    pub vertex_indirection_bytes: u64,
    pub position_bytes: u64,
    pub attribute_bytes: u64,
    pub vertex_data_bytes: u64,
    pub triangle_index_bytes: u64,
    pub total_bytes: u64,
    pub duplicate_vertex_bytes: u64,
    pub bytes_per_triangle: f64,
}

impl StorageMetrics {
    pub fn finalize(&mut self) {
        self.total_bytes = self.descriptor_bytes
            + self.vertex_indirection_bytes
            + self.vertex_data_bytes
            + self.triangle_index_bytes;
        self.bytes_per_triangle = ratio(self.total_bytes, self.triangle_count);
    }

    pub fn add(&mut self, other: &StorageMetrics) {
        self.cluster_count += other.cluster_count;
        self.vertex_count += other.vertex_count;
        self.triangle_count += other.triangle_count;
        self.descriptor_bytes += other.descriptor_bytes;
        self.vertex_indirection_bytes += other.vertex_indirection_bytes;
        self.position_bytes += other.position_bytes;
        self.attribute_bytes += other.attribute_bytes;
        self.vertex_data_bytes += other.vertex_data_bytes;
        self.triangle_index_bytes += other.triangle_index_bytes;
        self.total_bytes += other.total_bytes;
        self.duplicate_vertex_bytes += other.duplicate_vertex_bytes;
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PathReport {
    pub path: String,
    pub storage: StorageMetrics,
    pub memory: MemoryMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrimitiveReport {
    pub name: String,
    pub vertex_count: u64,
    pub triangle_count: u64,
    pub vertex_bytes: usize,
    pub cluster_count: u64,
    pub attributes: Vec<VertexAttribute>,
    pub standard_storage: StorageMetrics,
    pub localized_storage: StorageMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub input: String,
    pub cluster_config: ClusterConfig,
    pub cache_config: CacheConfig,
    pub wave_size: usize,
    pub cu_count: usize,
    pub waves_per_cu: usize,
    pub dram_bandwidth_gbps: f64,
    pub gpu_clock_ghz: f64,
    pub primitive_count: usize,
    pub total_vertices: u64,
    pub total_triangles: u64,
    pub total_clusters: u64,
    pub standard: PathReport,
    pub localized: PathReport,
    pub primitives: Vec<PrimitiveReport>,
    pub warnings: Vec<String>,
}

impl BenchmarkReport {
    pub fn print_human(&self, per_primitive: bool) {
        println!("meshbench: {}", self.input);
        println!(
            "geometry: {} primitives, {} vertices, {} triangles, {} clusters",
            self.primitive_count, self.total_vertices, self.total_triangles, self.total_clusters
        );
        println!(
            "cluster: max_vertices={} max_triangles={} cone_weight={:.3}",
            self.cluster_config.max_vertices,
            self.cluster_config.max_triangles,
            self.cluster_config.cone_weight
        );
        println!(
            "cache: line={}B sector={}B L1={}B/CU/{}way L2={}B/{}way latency L1={} L2={} DRAM={}",
            self.cache_config.line_size,
            self.cache_config.sector_size,
            self.cache_config.l1_size,
            self.cache_config.l1_ways,
            self.cache_config.l2_size,
            self.cache_config.l2_ways,
            self.cache_config.l1_latency,
            self.cache_config.l2_latency,
            self.cache_config.dram_latency
        );
        println!(
            "execution: CUs={} waves/CU={} total_waves={} wave_size={} DRAM={}GB/s clock={}GHz",
            self.cu_count,
            self.waves_per_cu,
            self.cu_count * self.waves_per_cu,
            self.wave_size,
            self.dram_bandwidth_gbps,
            self.gpu_clock_ghz
        );
        println!(
            "model: per-wave SIMD trace, dependency phases within each wave, independent waves in flight"
        );
        println!();

        println!(
            "{:<10} {:>12} {:>12} {:>9} {:>9} {:>12} {:>12} {:>7} {:>10} {:>12}",
            "path",
            "storage B",
            "logical B",
            "L1 hit%",
            "L2 hit%",
            "DRAM B",
            "DRAM sectors",
            "amp",
            "requests",
            "cycles"
        );
        self.print_path_row(&self.standard);
        self.print_path_row(&self.localized);
        println!(
            "waves: standard={} localized={}",
            self.standard.memory.waves, self.localized.memory.waves
        );
        println!(
            "wave groups: standard={} localized={}",
            self.standard.memory.wave_groups, self.localized.memory.wave_groups
        );
        println!(
            "bandwidth floor cycles: standard={} localized={}",
            self.standard.memory.bandwidth_latency_cycles,
            self.localized.memory.bandwidth_latency_cycles
        );
        println!(
            "effective DRAM bandwidth: standard={:.2} B/cycle localized={:.2} B/cycle",
            self.standard.memory.dram_bytes_per_cycle, self.localized.memory.dram_bytes_per_cycle
        );
        println!(
            "amp = DRAM bytes / shader-requested bytes; below 1.0 means requests coalesce into shared cache lines"
        );

        let storage_delta = delta(
            self.localized.storage.total_bytes,
            self.standard.storage.total_bytes,
        );
        let dram_delta = delta(
            self.localized.memory.dram_bytes,
            self.standard.memory.dram_bytes,
        );
        let cycle_delta = delta(
            self.localized.memory.estimated_latency_cycles,
            self.standard.memory.estimated_latency_cycles,
        );
        println!();
        println!(
            "localized vs standard: storage {:+.2}%, DRAM bytes {:+.2}%, cycles {:+.2}%",
            storage_delta, dram_delta, cycle_delta
        );
        println!(
            "bytes/triangle: standard logical={:.2} DRAM={:.2}, localized logical={:.2} DRAM={:.2}",
            self.standard.memory.logical_bytes_per_triangle,
            self.standard.memory.dram_bytes_per_triangle,
            self.localized.memory.logical_bytes_per_triangle,
            self.localized.memory.dram_bytes_per_triangle
        );

        println!();
        println!(
            "{:<10} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "path", "descriptors", "indirection", "positions", "attributes", "triangles"
        );
        self.print_breakdown_row(&self.standard);
        self.print_breakdown_row(&self.localized);

        if per_primitive {
            println!();
            println!(
                "{:<32} {:>10} {:>10} {:>9} {:>9} {:>12} {:>12}",
                "primitive",
                "vertices",
                "triangles",
                "vertex B",
                "clusters",
                "standard B",
                "localized B"
            );
            for primitive in &self.primitives {
                println!(
                    "{:<32} {:>10} {:>10} {:>9} {:>9} {:>12} {:>12}",
                    primitive.name,
                    primitive.vertex_count,
                    primitive.triangle_count,
                    primitive.vertex_bytes,
                    primitive.cluster_count,
                    primitive.standard_storage.total_bytes,
                    primitive.localized_storage.total_bytes
                );
                let layout = primitive
                    .attributes
                    .iter()
                    .map(|attribute| {
                        format!(
                            "{}:{}({}B)",
                            attribute.name, attribute.components, attribute.element_size
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("  layout: {layout}");
            }
        }

        if !self.warnings.is_empty() {
            println!();
            for warning in &self.warnings {
                eprintln!("warning: {warning}");
            }
        }
    }

    fn print_path_row(&self, path: &PathReport) {
        println!(
            "{:<10} {:>12} {:>12} {:>8.2}% {:>8.2}% {:>12} {:>12} {:>7.2} {:>10} {:>12}",
            path.path,
            path.storage.total_bytes,
            path.memory.logical_bytes,
            path.memory.l1_hit_rate * 100.0,
            path.memory.l2_hit_rate * 100.0,
            path.memory.dram_bytes,
            path.memory.dram_sectors,
            path.memory.line_amplification,
            path.memory.memory_requests,
            path.memory.estimated_latency_cycles
        );
    }

    fn print_breakdown_row(&self, path: &PathReport) {
        let breakdown = &path.memory.breakdown;
        println!(
            "{:<10} {:>12} {:>12} {:>12} {:>12} {:>12}",
            path.path,
            breakdown.descriptors.dram_bytes,
            breakdown.vertex_indirection.dram_bytes,
            breakdown.position_data.dram_bytes,
            breakdown.attribute_data.dram_bytes,
            breakdown.triangle_indices.dram_bytes
        );
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn delta(new: u64, old: u64) -> f64 {
    if old == 0 {
        0.0
    } else {
        (new as f64 - old as f64) / old as f64 * 100.0
    }
}
