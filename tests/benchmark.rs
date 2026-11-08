use meshbench::{BenchmarkOptions, CacheConfig, ClusterConfig, run};
use std::path::PathBuf;

#[test]
fn benchmark_box_gltf() {
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/assets/Box.glb");
    let options = BenchmarkOptions {
        cluster_config: ClusterConfig {
            max_vertices: 16,
            max_triangles: 4,
            cone_weight: 0.0,
        },
        cache_config: CacheConfig::default(),
        wave_size: 32,
        cu_count: 52,
        waves_per_cu: 40,
        dram_bandwidth_gbps: 560.0,
        gpu_clock_ghz: 1.825,
    };

    let report = run(&input, &options).expect("benchmark should succeed");

    assert!(report.primitive_count >= 1);
    assert!(report.total_vertices > 0);
    assert!(report.total_triangles > 0);
    assert!(report.total_clusters > 1);

    assert!(report.standard.storage.total_bytes > 0);
    assert!(report.localized.storage.total_bytes > 0);
    assert!(report.standard.memory.dram_bytes > 0);
    assert!(report.localized.memory.dram_bytes > 0);

    assert!(
        report
            .standard
            .memory
            .breakdown
            .vertex_indirection
            .dram_bytes
            > 0
    );
    assert_eq!(
        report
            .localized
            .memory
            .breakdown
            .vertex_indirection
            .dram_bytes,
        0
    );

    assert!(report.standard.memory.breakdown.position_data.dram_bytes > 0);
    assert!(report.standard.memory.breakdown.attribute_data.dram_bytes > 0);
    assert!(report.localized.memory.breakdown.position_data.dram_bytes > 0);
    assert!(report.localized.memory.breakdown.attribute_data.dram_bytes > 0);

    assert!(report.standard.memory.memory_requests > 0);
    assert!(report.localized.memory.memory_requests > 0);
    assert!(report.standard.memory.waves > 0);
    assert!(report.localized.memory.waves > 0);
    assert!(report.standard.memory.wave_groups > 0);
    assert!(report.localized.memory.wave_groups > 0);
    assert!(report.standard.memory.estimated_latency_cycles > 0);
    assert!(report.localized.memory.estimated_latency_cycles > 0);
    assert!(report.standard.memory.dram_sectors > 0);
    assert!(report.localized.memory.dram_sectors > 0);
    assert!(report.standard.memory.unique_sectors > 0);
    assert!(report.localized.memory.unique_sectors > 0);

    assert_eq!(
        report.standard.storage.position_bytes + report.standard.storage.attribute_bytes,
        report.standard.storage.vertex_data_bytes
    );
    assert_eq!(
        report.localized.storage.position_bytes + report.localized.storage.attribute_bytes,
        report.localized.storage.vertex_data_bytes
    );
}
