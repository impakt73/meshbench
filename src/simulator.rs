use crate::cache::{BufferAllocator, Cache, CacheConfig};
use crate::cluster::PrimitiveClusters;
use crate::geometry::PreparedPrimitive;
use crate::report::{LoadCategory, MemoryMetrics};
use std::collections::HashSet;
use std::mem::size_of;

pub fn simulate_standard(
    primitives: &[PreparedPrimitive],
    clusters: &[PrimitiveClusters],
    config: CacheConfig,
) -> MemoryMetrics {
    let descriptor_size = size_of::<meshopt::ffi::meshopt_Meshlet>() as u64;
    let mut memory = MemorySystem::new(config);
    let mut allocator = BufferAllocator::new(config.line_size);

    for (primitive, cluster) in primitives.iter().zip(clusters) {
        let standard = &cluster.standard;
        let descriptor_base = allocator.alloc(standard.meshlets.len() * descriptor_size as usize);
        let indirection_base = allocator.alloc(standard.vertices.len() * size_of::<u32>());
        let triangle_base = allocator.alloc(standard.triangles.len());
        let vertex_base = allocator.alloc(primitive.vertex_count * primitive.stride);

        for (index, meshlet) in standard.meshlets.iter().enumerate() {
            memory.load(
                descriptor_base + index as u64 * descriptor_size,
                descriptor_size,
                LoadCategory::Descriptor,
            );

            let vertex_offset = meshlet.vertex_offset as usize;
            let vertex_count = meshlet.vertex_count as usize;
            memory.load(
                indirection_base + vertex_offset as u64 * size_of::<u32>() as u64,
                vertex_count as u64 * size_of::<u32>() as u64,
                LoadCategory::VertexIndirection,
            );

            for local_vertex in 0..vertex_count {
                let global_vertex = standard.vertices[vertex_offset + local_vertex] as u64;
                memory.load(
                    vertex_base + global_vertex * primitive.stride as u64,
                    primitive.stride as u64,
                    LoadCategory::VertexData,
                );
            }

            memory.load(
                triangle_base + meshlet.triangle_offset as u64,
                meshlet.triangle_count as u64 * 3,
                LoadCategory::TriangleIndex,
            );
        }
    }

    memory.finish()
}

pub fn simulate_localized(
    primitives: &[PreparedPrimitive],
    clusters: &[PrimitiveClusters],
    config: CacheConfig,
) -> MemoryMetrics {
    let descriptor_size = size_of::<meshopt::ffi::meshopt_Meshlet>() as u64;
    let mut memory = MemorySystem::new(config);
    let mut allocator = BufferAllocator::new(config.line_size);

    for (primitive, cluster) in primitives.iter().zip(clusters) {
        let localized = &cluster.localized;
        let descriptor_base = allocator.alloc(localized.meshlets.len() * descriptor_size as usize);
        let vertex_base = allocator.alloc(localized.vertices.len());
        let triangle_base = allocator.alloc(localized.triangles.len());

        for (index, meshlet) in localized.meshlets.iter().enumerate() {
            memory.load(
                descriptor_base + index as u64 * descriptor_size,
                descriptor_size,
                LoadCategory::Descriptor,
            );

            memory.load(
                vertex_base + meshlet.vertex_offset as u64 * primitive.stride as u64,
                meshlet.vertex_count as u64 * primitive.stride as u64,
                LoadCategory::VertexData,
            );

            memory.load(
                triangle_base + meshlet.triangle_offset as u64,
                meshlet.triangle_count as u64 * 3,
                LoadCategory::TriangleIndex,
            );
        }
    }

    memory.finish()
}

struct MemorySystem {
    line_size: u64,
    l1: Cache,
    l2: Cache,
    config: CacheConfig,
    metrics: MemoryMetrics,
    unique_lines: HashSet<u64>,
}

impl MemorySystem {
    fn new(config: CacheConfig) -> Self {
        Self {
            line_size: config.line_size as u64,
            l1: Cache::new(config.l1_size, config.l1_ways, config.line_size),
            l2: Cache::new(config.l2_size, config.l2_ways, config.line_size),
            config,
            metrics: MemoryMetrics::default(),
            unique_lines: HashSet::new(),
        }
    }

    fn load(&mut self, address: u64, size: u64, category: LoadCategory) {
        if size == 0 {
            return;
        }

        self.metrics.logical_bytes += size;
        self.metrics.breakdown.get_mut(category).logical_bytes += size;

        let first_line = address / self.line_size;
        let last_line = (address + size - 1) / self.line_size;

        for line in first_line..=last_line {
            self.unique_lines.insert(line);
            self.metrics.l1_line_requests += 1;
            self.metrics.breakdown.get_mut(category).line_requests += 1;

            if self.l1.probe_and_fill(line) {
                self.metrics.l1_hits += 1;
                self.metrics.estimated_latency_cycles += u64::from(self.config.l1_latency);
            } else {
                self.metrics.l2_line_requests += 1;

                if self.l2.probe_and_fill(line) {
                    self.metrics.l2_hits += 1;
                    self.metrics.estimated_latency_cycles +=
                        u64::from(self.config.l1_latency + self.config.l2_latency);
                } else {
                    self.metrics.dram_lines += 1;
                    self.metrics.dram_bytes += self.line_size;
                    self.metrics.estimated_latency_cycles += u64::from(
                        self.config.l1_latency + self.config.l2_latency + self.config.dram_latency,
                    );

                    let category_metrics = self.metrics.breakdown.get_mut(category);
                    category_metrics.dram_lines += 1;
                    category_metrics.dram_bytes += self.line_size;
                }
            }
        }
    }

    fn finish(mut self) -> MemoryMetrics {
        self.metrics.unique_lines = self.unique_lines.len() as u64;
        self.metrics.finalize();
        self.metrics
    }
}
