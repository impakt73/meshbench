use anyhow::Result;
use clap::Parser;
use meshbench::{BenchmarkOptions, CacheConfig, ClusterConfig, run};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "meshbench",
    version,
    about = "System-level mesh shader memory simulator for clustered geometry"
)]
struct Cli {
    input: PathBuf,

    #[arg(long, default_value_t = 64)]
    max_vertices: usize,

    #[arg(long, default_value_t = 64)]
    max_triangles: usize,

    #[arg(long, default_value_t = 0.0)]
    cone_weight: f32,

    #[arg(long, default_value_t = 64)]
    cache_line_size: usize,

    #[arg(long, default_value_t = 32 * 1024)]
    l1_size: usize,

    #[arg(long, default_value_t = 4)]
    l1_ways: usize,

    #[arg(long, default_value_t = 512 * 1024)]
    l2_size: usize,

    #[arg(long, default_value_t = 8)]
    l2_ways: usize,

    #[arg(long, default_value_t = 32)]
    l1_latency: u32,

    #[arg(long, default_value_t = 200)]
    l2_latency: u32,

    #[arg(long, default_value_t = 600)]
    dram_latency: u32,

    #[arg(
        long,
        default_value_t = 32,
        help = "Lanes per SIMD wave used for clustered vertex processing"
    )]
    wave_size: usize,

    #[arg(
        long,
        default_value_t = 64,
        help = "Maximum independent wave requests in flight per execution phase"
    )]
    max_in_flight: usize,

    #[arg(long)]
    json: bool,

    #[arg(long)]
    per_primitive: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let options = BenchmarkOptions {
        cluster_config: ClusterConfig {
            max_vertices: cli.max_vertices,
            max_triangles: cli.max_triangles,
            cone_weight: cli.cone_weight,
        },
        cache_config: CacheConfig {
            line_size: cli.cache_line_size,
            l1_size: cli.l1_size,
            l1_ways: cli.l1_ways,
            l2_size: cli.l2_size,
            l2_ways: cli.l2_ways,
            l1_latency: cli.l1_latency,
            l2_latency: cli.l2_latency,
            dram_latency: cli.dram_latency,
        },
        wave_size: cli.wave_size,
        max_in_flight: cli.max_in_flight,
    };

    let report = run(&cli.input, &options)?;

    if cli.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        report.print_human(cli.per_primitive);
    }

    Ok(())
}
