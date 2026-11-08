use crate::cache::{BufferAllocator, Cache, CacheConfig};
use crate::cluster::PrimitiveClusters;
use crate::geometry::PreparedPrimitive;
use crate::report::{LoadCategory, MemoryMetrics};
use std::collections::HashSet;
use std::mem::size_of;

struct Request {
    lines: Vec<u64>,
    logical_bytes: u64,
    category: LoadCategory,
}

impl Request {
    fn from_range(base: u64, size: u64, line_size: u64, category: LoadCategory) -> Self {
        let mut lines = Vec::new();
        if size > 0 {
            let first = base / line_size;
            let last = (base + size - 1) / line_size;
            lines.extend(first..=last);
        }
        Self {
            lines,
            logical_bytes: size,
            category,
        }
    }

    fn from_lanes(addresses: &[u64], size: u64, line_size: u64, category: LoadCategory) -> Self {
        let mut lines = Vec::new();
        for &address in addresses {
            if size > 0 {
                let first = address / line_size;
                let last = (address + size - 1) / line_size;
                lines.extend(first..=last);
            }
        }
        lines.sort_unstable();
        lines.dedup();
        Self {
            lines,
            logical_bytes: addresses.len() as u64 * size,
            category,
        }
    }
}

pub fn simulate_standard(
    primitives: &[PreparedPrimitive],
    clusters: &[PrimitiveClusters],
    config: CacheConfig,
    wave_size: usize,
    max_in_flight: usize,
) -> MemoryMetrics {
    let descriptor_size = size_of::<meshopt::ffi::meshopt_Meshlet>() as u64;
    let line_size = config.line_size as u64;
    let mut memory = MemorySystem::new(config);
    let mut allocator = BufferAllocator::new(config.line_size);

    let mut descriptor_requests = Vec::new();
    let mut middle_phase = Vec::new();
    let mut vertex_requests = Vec::new();

    for (primitive, cluster) in primitives.iter().zip(clusters) {
        let standard = &cluster.standard;
        let max_vertices = cluster.max_vertices;
        let cluster_count = standard.meshlets.len();
        let descriptor_base = allocator.alloc(cluster_count * descriptor_size as usize);
        let indirection_base = allocator.alloc(cluster_count * max_vertices * size_of::<u32>());
        let triangle_base = allocator.alloc(standard.triangles.len());
        let stream_bases = primitive
            .streams
            .iter()
            .map(|stream| allocator.alloc(primitive.vertex_count * stream.element_size()))
            .collect::<Vec<_>>();

        for (index, meshlet) in standard.meshlets.iter().enumerate() {
            descriptor_requests.push(Request::from_range(
                descriptor_base + index as u64 * descriptor_size,
                descriptor_size,
                line_size,
                LoadCategory::Descriptor,
            ));

            let cluster_indirection_base =
                indirection_base + index as u64 * max_vertices as u64 * size_of::<u32>() as u64;
            for wave_start in (0..max_vertices).step_by(wave_size) {
                let lanes = (max_vertices - wave_start).min(wave_size);
                middle_phase.push(Request::from_range(
                    cluster_indirection_base + wave_start as u64 * size_of::<u32>() as u64,
                    lanes as u64 * size_of::<u32>() as u64,
                    line_size,
                    LoadCategory::VertexIndirection,
                ));
            }

            middle_phase.push(Request::from_range(
                triangle_base + meshlet.triangle_offset as u64,
                meshlet.triangle_count as u64 * 3,
                line_size,
                LoadCategory::TriangleIndex,
            ));

            let vertex_offset = meshlet.vertex_offset as usize;
            let actual_vertices = meshlet.vertex_count as usize;

            for (stream_index, stream) in primitive.streams.iter().enumerate() {
                let element_size = stream.element_size() as u64;
                let category = if stream_index == 0 {
                    LoadCategory::PositionData
                } else {
                    LoadCategory::AttributeData
                };

                for wave_start in (0..max_vertices).step_by(wave_size) {
                    let lanes = (max_vertices - wave_start).min(wave_size);
                    let mut addresses = Vec::with_capacity(lanes);

                    for slot in wave_start..wave_start + lanes {
                        let global_vertex = if actual_vertices == 0 {
                            0
                        } else if slot < actual_vertices {
                            standard.vertices[vertex_offset + slot] as u64
                        } else {
                            standard.vertices[vertex_offset + actual_vertices - 1] as u64
                        };
                        addresses.push(stream_bases[stream_index] + global_vertex * element_size);
                    }

                    vertex_requests.push(Request::from_lanes(
                        &addresses,
                        element_size,
                        line_size,
                        category,
                    ));
                }
            }
        }
    }

    memory.process_phase(&descriptor_requests, max_in_flight);
    memory.process_phase(&middle_phase, max_in_flight);
    memory.process_phase(&vertex_requests, max_in_flight);
    memory.finish()
}

pub fn simulate_localized(
    primitives: &[PreparedPrimitive],
    clusters: &[PrimitiveClusters],
    config: CacheConfig,
    wave_size: usize,
    max_in_flight: usize,
) -> MemoryMetrics {
    let descriptor_size = size_of::<meshopt::ffi::meshopt_Meshlet>() as u64;
    let line_size = config.line_size as u64;
    let mut memory = MemorySystem::new(config);
    let mut allocator = BufferAllocator::new(config.line_size);

    let mut descriptor_requests = Vec::new();
    let mut ready_phase = Vec::new();

    for (primitive, cluster) in primitives.iter().zip(clusters) {
        let localized = &cluster.localized;
        let max_vertices = cluster.max_vertices;
        let cluster_count = localized.len();
        let position_size = primitive.position().element_size() as u64;
        let descriptor_base = allocator.alloc(cluster_count * descriptor_size as usize);
        let position_base = allocator.alloc(cluster_count * max_vertices * position_size as usize);
        let attribute_bases = localized
            .attributes
            .iter()
            .map(|stream| allocator.alloc(cluster_count * max_vertices * stream.element_size()))
            .collect::<Vec<_>>();
        let triangle_base = allocator.alloc(localized.positions.triangles.len());

        for (index, meshlet) in localized.positions.meshlets.iter().enumerate() {
            descriptor_requests.push(Request::from_range(
                descriptor_base + index as u64 * descriptor_size,
                descriptor_size,
                line_size,
                LoadCategory::Descriptor,
            ));

            ready_phase.push(Request::from_range(
                triangle_base + meshlet.triangle_offset as u64,
                meshlet.triangle_count as u64 * 3,
                line_size,
                LoadCategory::TriangleIndex,
            ));

            let vertex_offset = meshlet.vertex_offset as usize;
            let actual_vertices = meshlet.vertex_count as usize;
            let cluster_position_base =
                position_base + index as u64 * max_vertices as u64 * position_size;

            for wave_start in (0..max_vertices).step_by(wave_size) {
                let lanes = (max_vertices - wave_start).min(wave_size);
                let mut addresses = Vec::with_capacity(lanes);

                for slot in wave_start..wave_start + lanes {
                    let local_slot = clamped_slot(slot, actual_vertices);
                    addresses.push(
                        cluster_position_base + (vertex_offset + local_slot) as u64 * position_size,
                    );
                }

                ready_phase.push(Request::from_lanes(
                    &addresses,
                    position_size,
                    line_size,
                    LoadCategory::PositionData,
                ));
            }

            for (stream_index, stream) in localized.attributes.iter().enumerate() {
                let element_size = stream.element_size() as u64;
                let cluster_attribute_base = attribute_bases[stream_index]
                    + index as u64 * max_vertices as u64 * element_size;

                for wave_start in (0..max_vertices).step_by(wave_size) {
                    let lanes = (max_vertices - wave_start).min(wave_size);
                    let mut addresses = Vec::with_capacity(lanes);

                    for slot in wave_start..wave_start + lanes {
                        let local_slot = clamped_slot(slot, actual_vertices);
                        addresses.push(
                            cluster_attribute_base
                                + (vertex_offset + local_slot) as u64 * element_size,
                        );
                    }

                    ready_phase.push(Request::from_lanes(
                        &addresses,
                        element_size,
                        line_size,
                        LoadCategory::AttributeData,
                    ));
                }
            }
        }
    }

    memory.process_phase(&descriptor_requests, max_in_flight);
    memory.process_phase(&ready_phase, max_in_flight);
    memory.finish()
}

fn clamped_slot(slot: usize, actual_vertices: usize) -> usize {
    if actual_vertices == 0 {
        0
    } else {
        slot.min(actual_vertices - 1)
    }
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

    fn process_phase(&mut self, requests: &[Request], max_in_flight: usize) {
        for chunk in requests.chunks(max_in_flight) {
            let mut chunk_latency = 0;

            for request in chunk {
                self.metrics.memory_requests += 1;
                self.metrics.logical_bytes += request.logical_bytes;
                self.metrics
                    .breakdown
                    .get_mut(request.category)
                    .logical_bytes += request.logical_bytes;

                let request_latency = self.process_request(request);
                chunk_latency = chunk_latency.max(request_latency);
            }

            self.metrics.phase_chunks += 1;
            self.metrics.estimated_latency_cycles += chunk_latency;
        }
    }

    fn process_request(&mut self, request: &Request) -> u64 {
        let mut latency = 0;

        for &line in &request.lines {
            self.unique_lines.insert(line);
            self.metrics.l1_line_requests += 1;
            self.metrics
                .breakdown
                .get_mut(request.category)
                .line_requests += 1;

            let line_latency = if self.l1.probe_and_fill(line) {
                self.metrics.l1_hits += 1;
                u64::from(self.config.l1_latency)
            } else {
                self.metrics.l2_line_requests += 1;

                if self.l2.probe_and_fill(line) {
                    self.metrics.l2_hits += 1;
                    u64::from(self.config.l1_latency + self.config.l2_latency)
                } else {
                    self.metrics.dram_lines += 1;
                    self.metrics.dram_bytes += self.line_size;

                    let category_metrics = self.metrics.breakdown.get_mut(request.category);
                    category_metrics.dram_lines += 1;
                    category_metrics.dram_bytes += self.line_size;

                    u64::from(
                        self.config.l1_latency + self.config.l2_latency + self.config.dram_latency,
                    )
                }
            };

            latency = latency.max(line_latency);
        }

        latency
    }

    fn finish(mut self) -> MemoryMetrics {
        self.metrics.unique_lines = self.unique_lines.len() as u64;
        self.metrics.finalize();
        self.metrics
    }
}
