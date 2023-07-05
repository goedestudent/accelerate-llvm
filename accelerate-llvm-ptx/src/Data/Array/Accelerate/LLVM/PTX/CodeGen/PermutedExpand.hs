{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE RebindableSyntax    #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE ViewPatterns        #-}
-- |
-- Module      : Data.Array.Accelerate.LLVM.PTX.CodeGen.PermutedExpand
-- Copyright   : [2016..2020] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.LLVM.PTX.CodeGen.PermutedExpand (

  mkPermutedExpand,

) where

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.ExpandFusionStrategy              ( ExpandFusionStrategy(..) )
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Elt
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Type

import Data.Array.Accelerate.LLVM.CodeGen.Arithmetic                as A hiding((-), negate)
import Data.Array.Accelerate.LLVM.CodeGen.Array
import Data.Array.Accelerate.LLVM.CodeGen.Base
import Data.Array.Accelerate.LLVM.CodeGen.Constant
import Data.Array.Accelerate.LLVM.CodeGen.Environment
import Data.Array.Accelerate.LLVM.CodeGen.Exp
import Data.Array.Accelerate.LLVM.CodeGen.IR
import Data.Array.Accelerate.LLVM.CodeGen.Loop                      (imapFromStepTo, iterFromStepTo, for, while)
import Data.Array.Accelerate.LLVM.CodeGen.Monad
import Data.Array.Accelerate.LLVM.CodeGen.Permute
import Data.Array.Accelerate.LLVM.CodeGen.Ptr
import Data.Array.Accelerate.LLVM.CodeGen.Sugar
import Data.Array.Accelerate.LLVM.Compile.Cache

import Data.Array.Accelerate.LLVM.PTX.CodeGen.Base                  as Base
import Data.Array.Accelerate.LLVM.PTX.CodeGen.Loop
import Data.Array.Accelerate.LLVM.PTX.CodeGen.Permute
import Data.Array.Accelerate.LLVM.PTX.Target

import LLVM.AST.Type.AddrSpace
import LLVM.AST.Type.Instruction
import LLVM.AST.Type.Instruction.Atomic
import LLVM.AST.Type.Instruction.RMW                                as RMW
import LLVM.AST.Type.Instruction.Volatile
import LLVM.AST.Type.Operand
import LLVM.AST.Type.Representation

import Foreign.CUDA.Analysis

import Control.Monad                                                ( void )
import Control.Monad.State                                          ( gets )
import Prelude
import Data.String                                                  (fromString)
import Data.Bits                                                    as Bits (shiftL)

----
-- Selection of loop/fusion strategy
----

-- | The type for functions that generate the outer loops for a kernel.
type PermutedExpandOuterLoop =
    forall sh sh' e e' aenv1 aenv2 env1 env2.
       IRArray (Array sh e)
    -> IRArray (Array DIM1 Int32)
    -> IROpenFun1 PTX env1 aenv1 (e -> Int)
    -> IROpenFun2 PTX env2 aenv2 (e -> Int -> (PrimMaybe sh', e'))
    -> TypeR e
    -> ShapeR sh
    -> (Operands (PrimMaybe sh') -> Operands e' -> CodeGen PTX ())
    -> CodeGen PTX ()

-- Defines what outer loop to use. Useful for benchmarking optimisations
permutedExpandOuter :: ExpandFusionStrategy -> PermutedExpandOuterLoop
permutedExpandOuter Basic = mkPermutedExpandOuterBasic -- Every thread handles 1 element of the input (expands and permutes it)
permutedExpandOuter Blocks = mkPermutedExpandOuterBlocks -- Highly inefficient. Every thread block handles 1 element of the input, every thread expands 1 element-index combination and permutes it. Every thread will read the input element to do this, thus every input element will be read multiple times.
permutedExpandOuter Shuffle = mkPermutedExpandOuterShuffle -- Every warp handles 1 element of the input, every lane expands 1 element-index combination and permutes it. A value is read only once per warp
permutedExpandOuter BlockShuffle = mkPermutedExpandOuterBlockShuffle -- Every thread block handles 1 element of the input. This element is read only once per warp. Every thread/lane expands 1 element-index combination and permutes it.
permutedExpandOuter SharedMem = mkPermutedExpandOuterSharedMem -- Every thread block handles 1 element of the input. The element is read only once per block and stored in shared memory. every thread expands 1 element-index combination and permutes it
permutedExpandOuter (MultiBlock i) = mkPermutedExpandOuterMultiBlock i True
permutedExpandOuter (MultiBlock' i) = mkPermutedExpandOuterMultiBlock i False


----
-- Generation of base kernel
----


-- Extension of forward permutation specified by an indexing mapping. The resulting array is
-- initialised with the given defaults, and any further values that are permuted
-- into the result array are added to the current value using the combination
-- function.
--
-- The combination function must be /associative/ and /commutative/. Elements
-- that are mapped to the magic index 'ignore' are dropped.
--
-- Parallel forward permutation has to take special care because different
-- threads could concurrently try to update the same memory location. Where
-- available we make use of special atomic instructions and other optimisations,
-- but in the general case each element of the output array has a lock which
-- must be obtained by the thread before it can update that memory location.
--
-- This version differs from a standard permutation, because it will also
-- perform an expansion of its input.
--
mkPermutedExpand
    :: HasCallStack
    => UID
    -> Gamma         aenv                     -- ^ Environment
    -> ArrayR              (Vector e)         -- ^ Input array of expand
    -> TypeR     e'                           -- ^ Result type after permutation
    -> ExpandFusionStrategy
    -> IRFun1    PTX aenv (e -> Int)          -- ^ Size function of expand
    -> IRFun2    PTX aenv (e -> Int -> (PrimMaybe sh', e')) -- ^ Get function of expand
    -> IRPermuteFun PTX aenv (e' -> e' -> e') -- ^ Combination function of permutation
    -> ShapeR    sh'                          -- ^ Output shape
    -> CodeGen   PTX      (IROpenAcc PTX aenv (Array sh' e'))
mkPermutedExpand uid aenv repr tp strategy sz get IRPermuteFun{..} shr' =
  case atomicRMW of
    Just (rmw, f) -> mkPermutedExpand_rmw   uid aenv repr tp strategy sz get rmw f   shr'
    _             -> mkPermutedExpand_mutex uid aenv repr tp strategy sz get combine shr'

-- Extension of parallel forward permutation function which uses atomic instructions to
-- implement lock-free array updates.
--
-- Atomic instruction support on CUDA devices is a bit patchy, so depending on
-- the element type and compute capability of the target hardware we may need to
-- emulate the operation using atomic compare-and-swap.
--
--              Int32    Int64    Float16    Float32    Float64
--           +-------------------------------------------------
--    (+)    |  2.0       2.0       7.0        2.0        6.0
--    (-)    |  2.0       2.0        x          x          x
--    (.&.)  |  2.0       3.2
--    (.|.)  |  2.0       3.2
--    xor    |  2.0       3.2
--    min    |  2.0       3.2        x          x          x
--    max    |  2.0       3.2        x          x          x
--    CAS    |  2.0       2.0
--
-- Note that NVPTX requires at least compute 2.0, so we can always implement the
-- lockfree update operations in terms of compare-and-swap.
--
-- This version differs from a standard permutation, because it will also
-- perform an expansion of its input.
--
mkPermutedExpand_rmw
    :: HasCallStack
    => UID
    -> Gamma         aenv                     -- ^ Environment
    -> ArrayR              (Vector e)         -- ^ Input array of expand
    -> TypeR     e'                           -- ^ Result type after permutation
    -> ExpandFusionStrategy
    -> IRFun1    PTX aenv (e -> Int)          -- ^ Size function of expand
    -> IRFun2    PTX aenv (e -> Int -> (PrimMaybe sh', e')) -- ^ Get function of expand
    -> RMWOperation                           -- ^ Combination function of permutation, is an atomic function
    -> IRFun1    PTX aenv (e' -> e')          -- ^ Function that makes sure that the new value (that is not in dfts array) has the correct index/environment
    -> ShapeR    sh'                          -- ^ Output shape
    -> CodeGen   PTX      (IROpenAcc PTX aenv (Array sh' e'))
mkPermutedExpand_rmw uid aenv repr@(ArrayR shr tp) tp' strategy sz get rmw update shr' = do
  dev <- liftCodeGen $ gets ptxDeviceProperties
  --
  let
      maxSzR              = ArrayR (ShapeRsnoc ShapeRz) (TupRsingle scalarTypeInt32)
      outR                = ArrayR shr' tp'
      (arrOut, paramOut)  = mutableArray outR "out"
      (arrMaxSz, paramMaxSz)= mutableArray maxSzR "maxSz"
      (arrIn,  paramIn)   = mutableArray repr "in"
      paramEnv            = envParam aenv
      --
      bytes               = bytesElt tp'
      compute             = computeCapability dev
      compute32           = Compute 3 2
      compute60           = Compute 6 0
      compute70           = Compute 7 0
  --
  makeOpenAcc uid "permutedexpand_rmw" (paramOut ++ paramMaxSz ++ paramIn ++ paramEnv) $ do
    (permutedExpandOuter strategy) arrIn arrMaxSz sz get tp shr $ \ix' x ->
      do
            j <- intOfIndex shr' (irArrayShape arrOut) =<< fromJust ix'-- Index to permute to
            r <- app1 update x

            case rmw of
              Exchange
                -> writeArray TypeInt arrOut j r
              --
              _ | TupRsingle (SingleScalarType s)   <- tp'                -- Get the actual SingleType
                , adata                             <- irArrayData arrOut -- Pointer to start of array in memory
                -> do
                      addr <- instr' $ GetElementPtr (asPtr defaultAddrSpace (op s adata)) [op integralType j] -- Address of element at index j of output array
                      --
                      let
                          rmw_integral :: IntegralType t -> Operand (Ptr t) -> Operand t -> CodeGen PTX ()
                          rmw_integral t ptr val
                            | primOk    = void . instr' $ AtomicRMW (IntegralNumType t) NonVolatile rmw ptr val (CrossThread, AcquireRelease)
                            | otherwise =
                                case rmw of
                                  RMW.And -> atomicCAS_rmw s' (A.band t (ir t val)) ptr
                                  RMW.Or  -> atomicCAS_rmw s' (A.bor  t (ir t val)) ptr
                                  RMW.Xor -> atomicCAS_rmw s' (A.xor  t (ir t val)) ptr
                                  RMW.Min -> atomicCAS_cmp s' A.lt ptr val
                                  RMW.Max -> atomicCAS_cmp s' A.gt ptr val
                                  _       -> internalError "unexpected transition"
                            where
                              s'      = NumSingleType (IntegralNumType t)
                              primOk  = compute >= compute32
                                    || bytes == 4
                                    || case rmw of
                                          RMW.Add -> True
                                          RMW.Sub -> True
                                          _       -> False

                          rmw_floating :: FloatingType t -> Operand (Ptr t) -> Operand t -> CodeGen PTX ()
                          rmw_floating t ptr val =
                            case rmw of
                              RMW.Min       -> atomicCAS_cmp s' A.lt ptr val
                              RMW.Max       -> atomicCAS_cmp s' A.gt ptr val
                              RMW.Sub       -> atomicCAS_rmw s' (A.sub n (ir t val)) ptr
                              RMW.Add
                                | primAdd   -> atomicAdd_f t ptr val
                                | otherwise -> atomicCAS_rmw s' (A.add n (ir t val)) ptr
                              _             -> internalError "unexpected transition"
                            where
                              n       = FloatingNumType t
                              s'      = NumSingleType n
                              primAdd =
                                case t of
                                  TypeHalf   -> compute >= compute70
                                  TypeFloat  -> True
                                  TypeDouble -> compute >= compute60
                      case s of
                        NumSingleType (IntegralNumType t) -> rmw_integral t addr (op t r)
                        NumSingleType (FloatingNumType t) -> rmw_floating t addr (op t r)
              --
              _ -> internalError "unexpected transition"

    return_

-- Extension of parallel forward permutation function which uses a spinlock to acquire
-- a mutex before updating the value at that location.
--
-- This version differs from a standard permutation, because it will also
-- perform an expansion of its input.
--
mkPermutedExpand_mutex
    :: UID
    -> Gamma         aenv                     -- ^ Environment
    -> ArrayR              (Vector e)         -- ^ Input array of expand
    -> TypeR     e'                           -- ^ Result type after permutation
    -> ExpandFusionStrategy
    -> IRFun1    PTX aenv (e -> Int)          -- ^ Size function of expand
    -> IRFun2    PTX aenv (e -> Int -> (PrimMaybe sh', e')) -- ^ Get function of expand
    -> IRFun2    PTX aenv (e' -> e' -> e')    -- ^ Combination function of permutation
    -> ShapeR    sh'                          -- ^ Output shape
    -> CodeGen   PTX      (IROpenAcc PTX aenv (Array sh' e'))
mkPermutedExpand_mutex uid aenv repr@(ArrayR shr tp) tp' strategy sz get combine shr' =
  let
      outR                  = ArrayR shr' tp'
      lockR                 = ArrayR (ShapeRsnoc ShapeRz) (TupRsingle scalarTypeWord32)
      maxSzR                = ArrayR (ShapeRsnoc ShapeRz) (TupRsingle scalarTypeInt32)
      (arrOut,  paramOut)   = mutableArray outR "out"
      (arrLock, paramLock)  = mutableArray lockR "lock"
      (arrMaxSz, paramMaxSz)= mutableArray maxSzR "maxSz"
      (arrIn,  paramIn)     = mutableArray repr "in"
      paramEnv              = envParam aenv
  in
  makeOpenAcc uid "permutedexpand_mutex" (paramOut ++ paramLock ++ paramMaxSz ++ paramIn ++ paramEnv) $ do

    (permutedExpandOuter strategy) arrIn arrMaxSz sz get tp shr $ \ix' x -> do
      -- project element onto the destination array and (atomically) update
      j <- intOfIndex shr' (irArrayShape arrOut) =<< fromJust ix'

      atomically arrLock j $ do
        y <- readArray TypeInt arrOut j
        r <- app2 combine x y
        writeArray TypeInt arrOut j r

    return_

----
-- Generation of outer loop using different strategies
----

mkPermutedExpandOuterBasic :: PermutedExpandOuterLoop
mkPermutedExpandOuterBasic arrIn _ sz get _ shr body = 
  let start = liftInt 0 
  in
    do
      let shIn  = irArrayShape arrIn  -- Shape of input array
      end       <- shapeSize shr shIn -- Size of input array

      -- for every element in the input
      imapFromTo start end $ \i -> do
        v    <- readArray TypeInt arrIn i
        end' <- app1 sz v -- How many elements this v expands to: sz v

        -- for i = 0 to sz v: permute (get v i)
        imapFromStepTo start (liftInt 1) end' $ \i' -> do
            tup <- app2 get v i'
            let ix' = A.fst tup -- Maybe index to permute to

            when (isJust ix') $ do
              let x = A.snd tup -- Element that we want to permute

              body ix' x

mkPermutedExpandOuterBlocks :: PermutedExpandOuterLoop
mkPermutedExpandOuterBlocks arrIn _ sz get _ shr body = 
  let start = liftInt 0
  in
    do
      let shIn  = irArrayShape arrIn  -- Shape of input array
      end       <- shapeSize shr shIn -- Size of input array

      -- For every element in the input, expand it and directly permute all values
      -- Note that imapFromTo automatically spreads it over the available threads
      imapFromToBlock start end $ \i -> do
        v    <- readArray TypeInt arrIn i 
        end' <- app1 sz v -- How many elements this v expands to: sz v

        -- for i = 0 to sz v: permute (get v i)
        --imapFromStepTo start (liftInt 1) end' $ \i' -> do
        imapFromToBlockThread start end' $ \i' -> do
            tup <- app2 get v i'
            let ix' = A.fst tup -- Maybe index to permute to

            when (isJust ix') $ do
              let x = A.snd tup -- Element that we want to permute

              body ix' x

mkPermutedExpandOuterShuffle :: PermutedExpandOuterLoop
mkPermutedExpandOuterShuffle arrIn _ sz get tp shr body = 
  let start = liftInt 0
  in
    do
      let shIn  = irArrayShape arrIn  -- Shape of input array
      end       <- shapeSize shr shIn -- Size of input array

      -- Every warp handles 1 item from the input
      imapFromToWarp start end $ \i -> do

        -- Read only if this is the first thread in the warp
        lane   <- laneId
        v'   <- if (tp, A.eq singleType lane (liftInt32 0)) 
                then readArray TypeInt arrIn i 
                else let go :: TypeR a -> Operands a
                         go TupRunit       = OP_Unit
                         go (TupRpair a b) = OP_Pair (go a) (go b)
                         go (TupRsingle t) = ir t (undef t)
                    in
                      return $ go tp
        __syncwarp

        v <- __shfl_idx tp v' (liftWord32 0)
        end' <- app1 sz v -- How many elements this v expands to: sz v

        -- Every thread will now expand and permute 1 item/index combination
        imapFromToLane start end' $ \i' -> do
            tup <- app2 get v i'
            let ix' = A.fst tup -- Maybe index to permute to

            when (isJust ix') $ do
              let x = A.snd tup -- Element that we want to permute

              body ix' x

mkPermutedExpandOuterBlockShuffle :: PermutedExpandOuterLoop
mkPermutedExpandOuterBlockShuffle arrIn _ sz get tp shr body = 
  let start = liftInt 0
  in
    do
      let shIn  = irArrayShape arrIn  -- Shape of input array
      end       <- shapeSize shr shIn -- Size of input array

      -- For every element in the input, expand it and directly permute all valuesn
      -- Note that imapFromTo automatically spreads it over the available threads
      imapFromToBlock start end $ \i -> do
        -- Read only if this is the first thread in the warp
        lane   <- laneId
        v'   <- if (tp, A.eq singleType lane (liftInt32 0)) 
                then readArray TypeInt arrIn i 
                else let go :: TypeR a -> Operands a
                         go TupRunit       = OP_Unit
                         go (TupRpair a b) = OP_Pair (go a) (go b)
                         go (TupRsingle t) = ir t (undef t)
                    in
                      return $ go tp
        __syncwarp

        v <- __shfl_idx tp v' (liftWord32 0)

        end' <- app1 sz v -- How many elements this v expands to: sz v

        -- for i = 0 to sz v: permute (get v i)
        --imapFromStepTo start (liftInt 1) end' $ \i' -> do
        imapFromToBlockThread start end' $ \i' -> do
            tup <- app2 get v i'
            let ix' = A.fst tup -- Maybe index to permute to

            when (isJust ix') $ do
              let x = A.snd tup -- Element that we want to permute

              body ix' x

mkPermutedExpandOuterSharedMem :: PermutedExpandOuterLoop
mkPermutedExpandOuterSharedMem arrIn _ sz get tp shr body = 
  let start = liftInt 0
  in
    do
      -- Shared memory is allocated per thread block, so all threads in the block have access to the same shared memory: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
      smem       <- staticSharedMem tp 1

      let shIn  = irArrayShape arrIn  -- Shape of input array
      end       <- shapeSize shr shIn -- Size of input array

      imapFromToBlock start end $ \i -> do
        tid   <- threadIdx

        when (A.eq singleType tid (liftInt32 0)) $ do
          v <-  readArray TypeInt arrIn i
          writeArray TypeInt smem (liftInt 0) v

        __syncthreads
        v <- readArray TypeInt smem (liftInt 0)

        end' <- app1 sz v -- How many elements this v expands to: sz v

        imapFromToBlockThread start end' $ \i' -> do
            tup <- app2 get v i'
            let ix' = A.fst tup -- Maybe index to permute to

            when (isJust ix') $ do
              let x = A.snd tup -- Element that we want to permute

              body ix' x

mkPermutedExpandOuterMultiBlock :: Int -> Bool -> PermutedExpandOuterLoop
mkPermutedExpandOuterMultiBlock multiBlockSizeI unrollBinarySearch arrIn _ sz get tp shr body = 
  let start = liftInt 0
      tpInt = TupRsingle $ SingleScalarType $ NumSingleType $ IntegralNumType $ TypeInt
      tpInt32 = TupRsingle $ SingleScalarType $ NumSingleType $ IntegralNumType $ TypeInt32
      multiBlockSizeW64 = Prelude.fromIntegral multiBlockSizeI :: Word64
      multiBlockSizeI32 = liftInt32 $ Prelude.fromIntegral (multiBlockSizeI + 1)
      multiBlockSize    = liftInt multiBlockSizeI
      multiBlockSize'   = liftInt (multiBlockSizeI - 1)
      binSearch
        | unrollBinarySearch = binSearchUnrolled multiBlockSizeI
        | otherwise          = binSearchRolled multiBlockSize
  in
    do
      -- Shared memory is allocated per thread block, so all threads in the block have access to the same shared memory: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
      smem       <- staticSharedMem tp multiBlockSizeW64
      smemSz     <- staticSharedMem tpInt multiBlockSizeW64

      let shIn  = irArrayShape arrIn  -- Shape of input array
      end       <- shapeSize shr shIn -- Size of input array
      tid       <- threadIdx

      imapFromStepToBlock start (multiBlockSize) end $ \inputBaseIdx -> do
        __syncthreads -- Make sure all threads in a block start at the same time.
                      -- This is to prevent threads that have finished early from starting to change data in shared memory, 
                      -- while other slower threads are still using it.
        imapFromToBlockThread start (multiBlockSize) $ \inputThreadOffset -> do
          inputIdx  <- add numType inputBaseIdx inputThreadOffset -- index of the element we are currently looking at

          -- writeArray TypeInt smemSz inputThreadOffset (liftInt 0)
          -- Make sure we do not read in elements that do not exist
          when (A.lt singleType inputIdx end) $ do 
            v    <- readArray TypeInt arrIn inputIdx
            size <- app1 sz v
            writeArray TypeInt smem   inputThreadOffset v
            writeArray TypeInt smemSz inputThreadOffset size

          -- If we are outside of bounds, just set size to 0. This also removes the need to check this in later computations
          when (A.gte singleType inputIdx end) $ do
            writeArray TypeInt smemSz inputThreadOffset (liftInt 0)

        __syncthreads

        -- Calculate prefix sum
        when (A.eq singleType tid (liftInt32 0)) $ do
          _ <- iterFromStepTo tpInt start (liftInt 1) (multiBlockSize) start $ \smemIdx pref -> do
            v  <- readArray TypeInt smemSz smemIdx
            v' <- add numType v pref
            writeArray TypeInt smemSz smemIdx v'
            return v'
          return_

        __syncthreads

        -- The total amount of elements expanded to by this block, is the result of the prefix sum
        end' <- readArray TypeInt smemSz (multiBlockSize')

        -- We perform 1 big loop: every thread gets an id from 0 to end' (globalGetIdx), and uses that to find both the relevant input element, and the index that should be used by 'get'
        imapFromToBlockThread start end' $ \globalGetIdx -> do
          i'       <- A.add numType globalGetIdx (liftInt 1) -- add 1, otherwise binary search is off-by-one
          smemIdx  <- binSearch i' smemSz                    -- get index of relevant input element
          pIdx     <- A.sub numType smemIdx (liftInt 1)      -- index of previous element
          pref     <- ifThenElse (tpInt, A.eq singleType smemIdx (liftInt 0)) (return $ liftInt 0) (readArray TypeInt smemSz pIdx) -- prefix sum of relevant element
          v        <- readArray TypeInt smem smemIdx -- read in relevant element
          idx'     <- A.sub numType globalGetIdx pref -- get expansion index
          tup      <- app2 get v idx' -- expand element

          let idx'' = A.fst tup -- Maybe index to permute to

          when (isJust idx'') $ do
            let x = A.snd tup -- Element that we want to permute

            body idx'' x
  where binSearch' target arr end = iterFromStepTo (TupRsingle $ SingleScalarType $ NumSingleType $ IntegralNumType $ TypeInt) (liftInt 0) (liftInt 1) end (liftInt 0)$ \i prevI -> do 
                                                                                                  i' <- A.add numType i (liftInt 1)
                                                                                                  ifThenElse (TupRsingle $ SingleScalarType $ NumSingleType $ IntegralNumType $ TypeInt, A.gt singleType target =<< readArray TypeInt arr i) (return i') (return prevI)

----
-- Utility functions used in kernel generation
----

-- For warp
imapFromToWarp :: Operands Int -> Operands Int -> (Operands Int -> CodeGen PTX ()) -> CodeGen PTX ()
imapFromToWarp start end body = do
  wsize  <- A.fromIntegral integralType numType =<< Base.warpSize -- # threads in warp
  bsize  <- blockDim -- # threads in block
  gsize  <- gridDim
  bwarps <- A.quot integralType bsize wsize -- # warps in a block

  step  <- A.fromIntegral integralType numType =<< A.mul numType bwarps gsize -- # total nr of warps (warps in block * blocks)
  wid   <- A.fromIntegral integralType numType =<< Base.warpId
  i0    <- add numType wid start
  --
  imapFromStepTo i0 step end body

-- For lane in warp
imapFromToLane :: Operands Int -> Operands Int -> (Operands Int -> CodeGen PTX ()) -> CodeGen PTX ()
imapFromToLane start end body = do
  step  <- A.fromIntegral integralType numType =<< Base.warpSize
  lid   <- A.fromIntegral integralType numType =<< laneId
  i0    <- add numType lid start
  --
  imapFromStepTo i0 step end body

-- For block
imapFromToBlock :: Operands Int -> Operands Int -> (Operands Int -> CodeGen PTX ()) -> CodeGen PTX ()
imapFromToBlock start end body = do
  step  <- A.fromIntegral integralType numType =<< gridDim
  bid   <- A.fromIntegral integralType numType =<< blockIdx
  i0    <- add numType bid start
  --
  imapFromStepTo i0 step end body

-- For block
imapFromStepToBlock :: Operands Int -> Operands Int -> Operands Int -> (Operands Int -> CodeGen PTX ()) -> CodeGen PTX ()
imapFromStepToBlock start step end body = do
  bStep <- A.fromIntegral integralType numType =<< gridDim
  step' <- mul numType bStep step
  bid   <- A.fromIntegral integralType numType =<< blockIdx
  i0    <- add numType bid start
  i0'   <- mul numType i0 step
  --
  imapFromStepTo i0' step' end body

-- For thread in block
imapFromToBlockThread :: Operands Int -> Operands Int -> (Operands Int -> CodeGen PTX ()) -> CodeGen PTX ()
imapFromToBlockThread start end body = do
  step  <- A.fromIntegral integralType numType =<< blockDim
  tid   <- A.fromIntegral integralType numType =<< threadIdx
  i0    <- add numType tid start
  --
  imapFromStepTo i0 step end body

  -- For thread in block
imapFromStepToBlockThread :: Operands Int -> Operands Int -> Operands Int -> (Operands Int -> CodeGen PTX ()) -> CodeGen PTX ()
imapFromStepToBlockThread start step end body = do
  tid   <- A.fromIntegral integralType numType =<< threadIdx
  i0    <- add numType tid start
  --
  imapFromStepTo i0 step end body

-- | Binary search, that looks for the index of a given value in the given array
binSearchRolled :: Operands Int               -- ^ Length of array
                -> Operands Int               -- ^ Target value to look for
                -> IRArray (Array DIM1 Int)   -- ^ Array to look for value for
                -> CodeGen PTX (Operands Int) -- ^ Return value: index of the first element that is greater than or equal to the target value
binSearchRolled end i arr = do
    right     <- A.sub numType end (liftInt 1)
    leftRight <- while intPairType (\(OP_Pair left right) -> gt singleType right left) body (pair (liftInt 0) right)
    return $ A.fst leftRight
  where intPairType = TupRpair typeInt typeInt
        typeInt     = TupRsingle $ SingleScalarType (NumSingleType (IntegralNumType TypeInt))
        body (OP_Pair left right) = do
                                      delta   <- add numType left right
                                      mid     <- A.quot integralType delta (liftInt 2)
                                      mid1    <- add numType mid (liftInt 1)
                                      value   <- readArray TypeInt arr mid
                                      ifThenElse (intPairType, lt singleType value i) (return $ pair mid1 right) (return $ pair left mid)

-- | Binary search, that looks for the index of a given value in the given array
binSearchUnrolled :: Int                        -- ^ Length of array
                  -> Operands Int               -- ^ Target value to search for
                  -> IRArray (Array DIM1 Int)   -- ^ Array in which to search for value
                  -> CodeGen PTX (Operands Int) -- ^ Return value: index of the first element that is greater than or equal to the target value
binSearchUnrolled arrayLength i arr = 
  let
    numAccBits = (Prelude.ceiling (Prelude.logBase 2 (Prelude.fromIntegral (arrayLength)))) - 1
    initialAcc = liftInt 0
    typeInt     = TupRsingle $ SingleScalarType (NumSingleType (IntegralNumType TypeInt))
  in
    do 
      target      <- A.sub numType i (liftInt 1)
      res         <- binSearch_loop numAccBits arr target initialAcc
      fstElem     <- readArray TypeInt arr (liftInt 0)
      resZero     <- A.eq singleType res (liftInt 0)
      fstLtTarget <- gt singleType fstElem target
      res'        <- ifThenElse (typeInt, land resZero fstLtTarget) (A.sub numType res (liftInt 1)) (return res)
      A.add numType res' (liftInt 1)

-- | Unrolled binary search, that looks for the index of a given value in the given array
binSearch_loop :: Int                        -- ^ Iteration of the unrolled loop
               -> IRArray (Array DIM1 Int)   -- ^ Array to look for value for
               -> Operands Int               -- ^ Target value to look for
               -> Operands Int               -- ^ Length of array
               -> CodeGen PTX (Operands Int) -- ^ Return value: index of the last element that is less than or equal to the target value
binSearch_loop (-1) arr target acc = return acc
binSearch_loop currentBit arr target acc = 
  let 
    mask        = liftInt $ 1 `Bits.shiftL` currentBit
    typeInt     = TupRsingle $ SingleScalarType (NumSingleType (IntegralNumType TypeInt))
  in do
    mid   <- A.add numType acc mask
    value <- readArray TypeInt arr mid
    acc'  <- ifThenElse (typeInt, lte singleType value target) (bor TypeInt acc mask) (return acc)
    binSearch_loop (currentBit - 1) arr target acc'

