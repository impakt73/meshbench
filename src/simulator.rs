use crate::cache::{BufferAllocator, Cache, CacheConfig};
use crate::cluster::PrimitiveClusters;
use crate::geometry::PreparedPrimitive;
use crate::report::{LoadCategory, MemoryMetrics};
use std::collections::HashSet;
use std::mem::size_of;

struct Request {
    sectors: Vec<u64>,
    logical_bytes: u64,
    category: LoadCategory,
}

impl Request {
    fn from_range(base: u64, size: u64, sector_size: u64, category: LoadCategory) -> Self {
        let mut sectors = Vec::new();
        if size > 0 {
            let first = base / sector_size;
            let last = (base + size - 1) / sector_size;
            sectors.extend(first..=last);
        }
        Self {
            sectors,
            logical_bytes: size,
            category,
        }
    }

    fn from_lanes(addresses: &[u64], size: u64, sector_size: u64, category: LoadCategory) -> Self {
        let mut sectors = Vec::new();
        for &address in addresses {
            if size > 0 {
                let first = address / sector_size;
                let last = (address + size - 1) / sector_size;
                sectors.extend(first..=last);
            }
        }
        sectors.sort_unstable();
        sectors.dedup();
        Self {
            sectors,
            logical_bytes: addresses.len() as u64 * size,
            category,
        }
    }
}

struct Wave {
    phases: Vec<Vec<Request>>,
}

pub fn simulate_standard(
    primitives: &[PreparedPrimitive],
    clusters: &[PrimitiveClusters],
    config: CacheConfig,
    wave_size: usize,
    cu_count: usize,
    waves_per_cu: usize,
    bandwidth_bytes_per_cycle: f64,
) -> MemoryMetrics {
    let descriptor_size = size_of::<meshopt::ffi::meshopt_Meshlet>() as u64;
    let sector_size = config.sector_size as u64;
    let mut memory = MemorySystem::new(config, cu_count);
    let mut allocator = BufferAllocator::new(config.line_size);
    let mut clusters_waves = Vec::new();

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
            let mut cluster_waves = Vec::new();
            let cluster_indirection_base =
                indirection_base + index as u64 * max_vertices as u64 * size_of::<u32>() as u64;
            let vertex_offset = meshlet.vertex_offset as usize;
            let actual_vertices = meshlet.vertex_count as usize;
            let wave_count = max_vertices.div_ceil(wave_size);

            for wave_index in 0..wave_count {
                let wave_start = wave_index * wave_size;
                let lanes = (max_vertices - wave_start).min(wave_size);
                let mut phases = Vec::new();
                let mut metadata_phase = Vec::new();

                if wave_index == 0 {
                    metadata_phase.push(Request::from_range(
                        descriptor_base + index as u64 * descriptor_size,
                        descriptor_size,
                        sector_size,
                        LoadCategory::Descriptor,
                    ));
                    metadata_phase.push(Request::from_range(
                        triangle_base + meshlet.triangle_offset as u64,
                        meshlet.triangle_count as u64 * 3,
                        sector_size,
                        LoadCategory::TriangleIndex,
                    ));
                }

                metadata_phase.push(Request::from_range(
                    cluster_indirection_base + wave_start as u64 * size_of::<u32>() as u64,
                    lanes as u64 * size_of::<u32>() as u64,
                    sector_size,
                    LoadCategory::VertexIndirection,
                ));
                phases.push(metadata_phase);

                let mut vertex_phase = Vec::new();
                for (stream_index, stream) in primitive.streams.iter().enumerate() {
                    let element_size = stream.element_size() as u64;
                    let category = if stream_index == 0 {
                        LoadCategory::PositionData
                    } else {
                        LoadCategory::AttributeData
                    };
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

                    vertex_phase.push(Request::from_lanes(
                        &addresses,
                        element_size,
                        sector_size,
                        category,
                    ));
                }
                phases.push(vertex_phase);
                cluster_waves.push(Wave { phases });
            }
            clusters_waves.push(cluster_waves);
        }
    }

    memory.process_clusters(
        &clusters_waves,
        cu_count,
        waves_per_cu,
        bandwidth_bytes_per_cycle,
    );
    memory.finish()
}

pub fn simulate_localized(
    primitives: &[PreparedPrimitive],
    clusters: &[PrimitiveClusters],
    config: CacheConfig,
    wave_size: usize,
    cu_count: usize,
    waves_per_cu: usize,
    bandwidth_bytes_per_cycle: f64,
) -> MemoryMetrics {
    let descriptor_size = size_of::<meshopt::ffi::meshopt_Meshlet>() as u64;
    let sector_size = config.sector_size as u64;
    let mut memory = MemorySystem::new(config, cu_count);
    let mut allocator = BufferAllocator::new(config.line_size);
    let mut clusters_waves = Vec::new();

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
            let mut cluster_waves = Vec::new();
            let vertex_offset = meshlet.vertex_offset as usize;
            let actual_vertices = meshlet.vertex_count as usize;
            let cluster_position_base =
                position_base + index as u64 * max_vertices as u64 * position_size;
            let wave_count = max_vertices.div_ceil(wave_size);

            for wave_index in 0..wave_count {
                let wave_start = wave_index * wave_size;
                let lanes = (max_vertices - wave_start).min(wave_size);
                let mut phase = Vec::new();

                if wave_index == 0 {
                    phase.push(Request::from_range(
                        descriptor_base + index as u64 * descriptor_size,
                        descriptor_size,
                        sector_size,
                        LoadCategory::Descriptor,
                    ));
                    phase.push(Request::from_range(
                        triangle_base + meshlet.triangle_offset as u64,
                        meshlet.triangle_count as u64 * 3,
                        sector_size,
                        LoadCategory::TriangleIndex,
                    ));
                }

                let mut addresses = Vec::with_capacity(lanes);
                for slot in wave_start..wave_start + lanes {
                    let local_slot = clamped_slot(slot, actual_vertices);
                    addresses.push(
                        cluster_position_base + (vertex_offset + local_slot) as u64 * position_size,
                    );
                }
                phase.push(Request::from_lanes(
                    &addresses,
                    position_size,
                    sector_size,
                    LoadCategory::PositionData,
                ));

                for (stream_index, stream) in localized.attributes.iter().enumerate() {
                    let element_size = stream.element_size() as u64;
                    let cluster_attribute_base = attribute_bases[stream_index]
                        + index as u64 * max_vertices as u64 * element_size;
                    let mut addresses = Vec::with_capacity(lanes);

                    for slot in wave_start..wave_start + lanes {
                        let local_slot = clamped_slot(slot, actual_vertices);
                        addresses.push(
                            cluster_attribute_base
                                + (vertex_offset + local_slot) as u64 * element_size,
                        );
                    }

                    phase.push(Request::from_lanes(
                        &addresses,
                        element_size,
                        sector_size,
                        LoadCategory::AttributeData,
                    ));
                }

                cluster_waves.push(Wave {
                    phases: vec![phase],
                });
            }
            clusters_waves.push(cluster_waves);
        }
    }

    memory.process_clusters(
        &clusters_waves,
        cu_count,
        waves_per_cu,
        bandwidth_bytes_per_cycle,
    );
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
    sector_size: u64,
    sectors_per_line: u64,
    l1_caches: Vec<Cache>,
    l2: Cache,
    config: CacheConfig,
    metrics: MemoryMetrics,
    unique_lines: HashSet<u64>,
    unique_sectors: HashSet<u64>,
    dram_lines: HashSet<u64>,
}

impl MemorySystem {
    fn new(config: CacheConfig, cu_count: usize) -> Self {
        let l1_caches = (0..cu_count)
            .map(|_| {
                Cache::new(
                    config.l1_size,
                    config.l1_ways,
                    config.line_size,
                    config.sector_size,
                )
            })
            .collect();
        Self {
            sector_size: config.sector_size as u64,
            sectors_per_line: (config.line_size / config.sector_size) as u64,
            l1_caches,
            l2: Cache::new(
                config.l2_size,
                config.l2_ways,
                config.line_size,
                config.sector_size,
            ),
            config,
            metrics: MemoryMetrics::default(),
            unique_lines: HashSet::new(),
            unique_sectors: HashSet::new(),
            dram_lines: HashSet::new(),
        }
    }

    fn process_clusters(
        &mut self,
        clusters_waves: &[Vec<Wave>],
        cu_count: usize,
        waves_per_cu: usize,
        bandwidth_bytes_per_cycle: f64,
    ) {
        let total_waves = clusters_waves.iter().map(Vec::len).sum::<usize>();
        self.metrics.waves += total_waves as u64;

        let mut per_cu: Vec<Vec<&Wave>> = vec![Vec::new(); cu_count];
        for (cluster_index, cluster_waves) in clusters_waves.iter().enumerate() {
            let cu = cluster_index % cu_count;
            per_cu[cu].extend(cluster_waves.iter());
        }

        let max_batches = per_cu
            .iter()
            .map(|waves| waves.len().div_ceil(waves_per_cu))
            .max()
            .unwrap_or(0);

        for batch in 0..max_batches {
            let mut batch_latency = 0;
            let dram_before = self.metrics.dram_bytes;

            for (cu, waves) in per_cu.iter().enumerate() {
                let start = batch * waves_per_cu;
                if start >= waves.len() {
                    continue;
                }
                let end = (start + waves_per_cu).min(waves.len());
                let mut cu_latency = 0;

                for wave in &waves[start..end] {
                    let wave_latency = self.process_wave(cu, wave);
                    cu_latency = cu_latency.max(wave_latency);
                }

                batch_latency = batch_latency.max(cu_latency);
            }

            let batch_dram_bytes = self.metrics.dram_bytes - dram_before;
            let bandwidth_latency = if bandwidth_bytes_per_cycle > 0.0 {
                (batch_dram_bytes as f64 / bandwidth_bytes_per_cycle).ceil() as u64
            } else {
                0
            };

            self.metrics.bandwidth_latency_cycles += bandwidth_latency;
            batch_latency = batch_latency.max(bandwidth_latency);
            self.metrics.wave_groups += 1;
            self.metrics.estimated_latency_cycles += batch_latency;
        }
    }

    fn process_wave(&mut self, cu: usize, wave: &Wave) -> u64 {
        let mut latency = 0;

        for phase in &wave.phases {
            latency += self.process_phase(cu, phase);
        }

        latency
    }

    fn process_phase(&mut self, cu: usize, requests: &[Request]) -> u64 {
        let mut phase_latency = 0;

        for request in requests {
            self.metrics.memory_requests += 1;
            self.metrics.logical_bytes += request.logical_bytes;
            self.metrics
                .breakdown
                .get_mut(request.category)
                .logical_bytes += request.logical_bytes;

            let request_latency = self.process_request(cu, request);
            phase_latency = phase_latency.max(request_latency);
        }

        phase_latency
    }

    fn process_request(&mut self, cu: usize, request: &Request) -> u64 {
        let MemorySystem {
            l1_caches,
            l2,
            config,
            metrics,
            unique_lines,
            unique_sectors,
            dram_lines,
            sectors_per_line,
            sector_size,
            ..
        } = self;
        let l1 = &mut l1_caches[cu];
        let mut latency = 0;

        for &sector in &request.sectors {
            let line = sector / *sectors_per_line;
            unique_sectors.insert(sector);
            unique_lines.insert(line);
            metrics.l1_sector_requests += 1;
            metrics.breakdown.get_mut(request.category).sector_requests += 1;

            let sector_latency = if l1.access_sector(sector) {
                metrics.l1_hits += 1;
                u64::from(config.l1_latency)
            } else {
                metrics.l2_sector_requests += 1;

                if l2.access_sector(sector) {
                    metrics.l2_hits += 1;
                    u64::from(config.l1_latency + config.l2_latency)
                } else {
                    metrics.dram_sectors += 1;
                    metrics.dram_bytes += *sector_size;
                    dram_lines.insert(line);

                    let category_metrics = metrics.breakdown.get_mut(request.category);
                    category_metrics.dram_sectors += 1;
                    category_metrics.dram_bytes += *sector_size;

                    u64::from(config.l1_latency + config.l2_latency + config.dram_latency)
                }
            };

            latency = latency.max(sector_latency);
        }

        latency
    }

    fn finish(mut self) -> MemoryMetrics {
        self.metrics.unique_lines = self.unique_lines.len() as u64;
        self.metrics.unique_sectors = self.unique_sectors.len() as u64;
        self.metrics.dram_lines = self.dram_lines.len() as u64;
        self.metrics.finalize();
        self.metrics
    }
}
