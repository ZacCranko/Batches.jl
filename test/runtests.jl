using Base.Test, Batches

@testset "Testing stuff" begin
    b1 = batches(1:100, 5)
    b2 = batches((1:100, 1:100), 5)
    b3 = batches(collect(1:100), 5, prealloc=true)
    b4 = batches((collect(1:100), collect(1:100)), 5, prealloc=true)

    @test b1[1] == 1:5
    @test b1[2] == 6:10
    @test b2[1] == (1:5, 1:5)
    @test length(b1)  == 20
    
    @test_throws ErrorException batches(1:100, 101)
    @test_throws BoundsError b1[length(b1) + 1]
    @test_throws DimensionMismatch batches((1:10_00,1:10_101), 100)

    @test begin # test iteration
        for b in b1
            b
        end 
        true
    end
end