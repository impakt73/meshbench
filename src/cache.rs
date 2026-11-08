use anyhow::{Result, bail};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CacheConfig {
    pub line_size: usize,
    pub sector_size: usize,
    pub l1_size: usize,
    pub l1_ways: usize,
    pub l2_size: usize,
    pub l2_ways: usize,
    pub l1_latency: u32,
    pub l2_latency: u32,
    pub dram_latency: u32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            line_size: 128,
            sector_size: 32,
            l1_size: 128 * 1024,
            l1_ways: 4,
            l2_size: 5 * 1024 * 1024,
            l2_ways: 8,
            l1_latency: 32,
            l2_latency: 200,
            dram_latency: 600,
        }
    }
}

impl CacheConfig {
    pub fn validate(&self) -> Result<()> {
        if self.line_size == 0 || !self.line_size.is_power_of_two() {
            bail!("cache line size must be a non-zero power of two");
        }
        if self.sector_size == 0 || !self.sector_size.is_power_of_two() {
            bail!("cache sector size must be a non-zero power of two");
        }
        if self.sector_size > self.line_size || !self.line_size.is_multiple_of(self.sector_size) {
            bail!("cache sector size must divide the cache line size");
        }
        if self.l1_ways == 0 || self.l2_ways == 0 {
            bail!("cache associativity must be greater than zero");
        }
        if self.l1_size < self.line_size * self.l1_ways {
            bail!("L1 cache must be at least one line per way");
        }
        if self.l2_size < self.line_size * self.l2_ways {
            bail!("L2 cache must be at least one line per way");
        }
        if self.l2_size < self.l1_size {
            bail!("L2 cache must not be smaller than L1");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct CacheLine {
    tag: u64,
    age: u64,
    sectors: u64,
}

pub struct Cache {
    sets: u64,
    sectors_per_line: u64,
    clock: u64,
    data: Vec<Vec<Option<CacheLine>>>,
}

impl Cache {
    pub fn new(size_bytes: usize, ways: usize, line_size: usize, sector_size: usize) -> Self {
        let line_size = line_size as u64;
        let sector_size = sector_size as u64;
        let sectors_per_line = line_size / sector_size;
        let sets = ((size_bytes as u64) / (line_size * ways as u64)).max(1);
        Self {
            sets,
            sectors_per_line,
            clock: 0,
            data: vec![vec![None; ways]; sets as usize],
        }
    }

    pub fn access_sector(&mut self, sector: u64) -> bool {
        let line = sector / self.sectors_per_line;
        let sector_bit = 1u64 << (sector % self.sectors_per_line);
        let set_index = (line % self.sets) as usize;
        let tag = line / self.sets;
        self.clock += 1;
        let age = self.clock;
        let set = &mut self.data[set_index];

        for cached in set.iter_mut().flatten() {
            if cached.tag == tag {
                let hit = cached.sectors & sector_bit != 0;
                cached.sectors |= sector_bit;
                cached.age = age;
                return hit;
            }
        }

        let victim = set
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry.as_ref().map_or(0, |line| line.age))
            .map(|(index, _)| index)
            .unwrap_or(0);

        set[victim] = Some(CacheLine {
            tag,
            age,
            sectors: sector_bit,
        });
        false
    }
}

pub struct BufferAllocator {
    next: u64,
    line_size: u64,
}

impl BufferAllocator {
    pub fn new(line_size: usize) -> Self {
        Self {
            next: 0x1000,
            line_size: line_size as u64,
        }
    }

    pub fn alloc(&mut self, size: usize) -> u64 {
        let base = align_up(self.next, self.line_size);
        let size = align_up(size as u64, self.line_size);
        self.next = base + size + self.line_size;
        base
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_sector_is_hit_after_fill() {
        let mut cache = Cache::new(4096, 2, 128, 32);
        assert!(!cache.access_sector(10));
        assert!(cache.access_sector(10));
    }

    #[test]
    fn cache_line_sectors_fill_independently() {
        let mut cache = Cache::new(4096, 2, 128, 32);
        assert!(!cache.access_sector(0));
        assert!(!cache.access_sector(1));
        assert!(cache.access_sector(0));
        assert!(cache.access_sector(1));
    }

    #[test]
    fn cache_evicts_least_recently_used_line() {
        let mut cache = Cache::new(256, 2, 128, 32);
        assert!(!cache.access_sector(0));
        assert!(!cache.access_sector(4));
        assert!(cache.access_sector(0));
        assert!(!cache.access_sector(8));
        assert!(!cache.access_sector(4));
    }

    #[test]
    fn allocator_returns_line_aligned_addresses() {
        let mut allocator = BufferAllocator::new(128);
        let first = allocator.alloc(3);
        let second = allocator.alloc(129);
        assert_eq!(first % 128, 0);
        assert_eq!(second % 128, 0);
        assert!(second > first);
    }
}
