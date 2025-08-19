# frozen_string_literal: true

require 'spec_helper'

RSpec.describe ClusterKit::Dimensionality::UMAP do
  describe 'seed behavior for reproducibility and performance' do
    let(:test_data) do
      # Generate deterministic test data
      rng = Random.new(42)
      50.times.map do
        10.times.map { rng.rand }
      end
    end

    context 'with seed (reproducible but slower)' do
      it 'produces identical results across multiple runs' do
        seed = 12345
        
        # Run 1
        umap1 = described_class.new(n_components: 2, n_neighbors: 5, random_seed: seed)
        result1 = umap1.fit_transform(test_data)
        
        # Run 2
        umap2 = described_class.new(n_components: 2, n_neighbors: 5, random_seed: seed)
        result2 = umap2.fit_transform(test_data)
        
        # Results should be identical
        expect(result1.size).to eq(result2.size)
        result1.zip(result2).each do |row1, row2|
          row1.zip(row2).each do |val1, val2|
            expect(val1).to be_within(1e-10).of(val2)
          end
        end
      end

      it 'produces different results with different seeds' do
        umap1 = described_class.new(n_components: 2, n_neighbors: 5, random_seed: 111)
        result1 = umap1.fit_transform(test_data)
        
        umap2 = described_class.new(n_components: 2, n_neighbors: 5, random_seed: 222)
        result2 = umap2.fit_transform(test_data)
        
        # Results should be different
        differences = result1.zip(result2).map do |row1, row2|
          row1.zip(row2).map { |v1, v2| (v1 - v2).abs }.sum
        end.sum
        
        expect(differences).to be > 0.1
      end
    end

    context 'without seed (fast but non-reproducible)' do
      it 'produces valid results' do
        umap = described_class.new(n_components: 2, n_neighbors: 5)
        result = umap.fit_transform(test_data)
        
        expect(result).to be_an(Array)
        expect(result.size).to eq(test_data.size)
        expect(result.first.size).to eq(2)
        
        # All values should be finite
        result.each do |row|
          row.each do |val|
            expect(val).to be_finite
          end
        end
      end
      
      it 'may produce different results across runs (non-deterministic)' do
        # Note: This test may occasionally fail by chance if the random
        # initialization happens to be the same
        umap1 = described_class.new(n_components: 2, n_neighbors: 5, random_seed: nil)
        result1 = umap1.fit_transform(test_data)
        
        umap2 = described_class.new(n_components: 2, n_neighbors: 5, random_seed: nil)
        result2 = umap2.fit_transform(test_data)
        
        # We can't guarantee they're different (might be same by chance),
        # but we can verify both are valid
        expect(result1).to be_an(Array)
        expect(result2).to be_an(Array)
        expect(result1.size).to eq(test_data.size)
        expect(result2.size).to eq(test_data.size)
      end
    end

    describe 'performance characteristics' do
      let(:large_data) do
        # Larger dataset to show performance difference
        rng = Random.new(42)
        500.times.map do
          20.times.map { rng.rand }
        end
      end

      it 'documents performance trade-offs' do
        require 'benchmark'
        
        # Time with seed (reproducible, uses serial_insert)
        time_with_seed = Benchmark.realtime do
          umap = described_class.new(n_components: 2, n_neighbors: 10, random_seed: 123)
          umap.fit_transform(large_data)
        end
        
        # Time without seed (fast, uses parallel_insert)
        time_without_seed = Benchmark.realtime do
          umap = described_class.new(n_components: 2, n_neighbors: 10, random_seed: nil)
          umap.fit_transform(large_data)
        end
        
        puts "\n  Performance comparison:"
        puts "    With seed (serial):    #{(time_with_seed * 1000).round(2)}ms"
        puts "    Without seed (parallel): #{(time_without_seed * 1000).round(2)}ms"
        
        # Without seed should generally be faster (though not guaranteed on small datasets)
        # We don't assert this as it depends on hardware and data size
      end
    end

    describe 'fit and transform separately' do
      context 'with seed' do
        it 'produces consistent transforms' do
          seed = 999
          umap = described_class.new(n_components: 2, n_neighbors: 5, random_seed: seed)
          
          # Fit on training data
          umap.fit(test_data)
          
          # Transform should produce consistent results
          transform1 = umap.transform(test_data)
          transform2 = umap.transform(test_data)
          
          transform1.zip(transform2).each do |row1, row2|
            row1.zip(row2).each do |val1, val2|
              expect(val1).to be_within(1e-10).of(val2)
            end
          end
        end
        
        it 'multiple fit_transform calls with same seed produce identical results' do
          seed = 888
          
          # First UMAP with fit_transform
          umap1 = described_class.new(n_components: 2, n_neighbors: 5, random_seed: seed)
          result1 = umap1.fit_transform(test_data)
          
          # Second UMAP with fit_transform and same seed
          umap2 = described_class.new(n_components: 2, n_neighbors: 5, random_seed: seed)
          result2 = umap2.fit_transform(test_data)
          
          # Results should be identical
          result1.zip(result2).each do |row1, row2|
            row1.zip(row2).each do |val1, val2|
              expect(val1).to be_within(1e-10).of(val2)
            end
          end
        end
        
        it 'transform approximates the training embedding (not identical)' do
          # This test documents that transform uses k-NN approximation
          # and won't produce identical results to the training embedding
          seed = 777
          
          umap = described_class.new(n_components: 2, n_neighbors: 5, random_seed: seed)
          embedded = umap.fit_transform(test_data)
          transformed = umap.transform(test_data)
          
          # Results should be similar but not identical
          # Transform uses k-NN approximation which is different from the actual embedding
          differences = embedded.zip(transformed).map do |row1, row2|
            row1.zip(row2).map { |v1, v2| (v1 - v2).abs }.sum
          end
          
          # Should be close but not exact
          avg_diff = differences.sum / differences.size
          expect(avg_diff).to be > 0.0001  # Not identical
          expect(avg_diff).to be < 1.0     # But still reasonably close
        end
      end
      
      context 'without seed' do
        it 'can still transform new data consistently after fit' do
          umap = described_class.new(n_components: 2, n_neighbors: 5)
          umap.fit(test_data)
          
          # Even without seed, transform should be consistent once fitted
          transform1 = umap.transform(test_data[0..9])
          transform2 = umap.transform(test_data[0..9])
          
          transform1.zip(transform2).each do |row1, row2|
            row1.zip(row2).each do |val1, val2|
              expect(val1).to be_within(1e-10).of(val2)
            end
          end
        end
      end
    end
    
    describe 'seed parameter validation' do
      it 'accepts nil seed' do
        umap = described_class.new(random_seed: nil)
        expect(umap.random_seed).to be_nil
      end
      
      it 'accepts integer seed' do
        umap = described_class.new(random_seed: 12345)
        expect(umap.random_seed).to eq(12345)
      end
      
      it 'accepts zero as seed' do
        umap = described_class.new(random_seed: 0)
        expect(umap.random_seed).to eq(0)
      end
      
      it 'produces reproducible results with seed 0' do
        umap1 = described_class.new(n_components: 2, n_neighbors: 5, random_seed: 0)
        result1 = umap1.fit_transform(test_data)
        
        umap2 = described_class.new(n_components: 2, n_neighbors: 5, random_seed: 0)
        result2 = umap2.fit_transform(test_data)
        
        result1.zip(result2).each do |row1, row2|
          row1.zip(row2).each do |val1, val2|
            expect(val1).to be_within(1e-10).of(val2)
          end
        end
      end
    end
  end
end