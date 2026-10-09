#!/bin/sh
# Build the Fortran AIOMFAC timing driver used by ../fortran_bench.py.
# usage: sh build.sh <AIOMFAC/FortranCode> <work_dir>
#   AIOMFAC/FortranCode: andizuend/AIOMFAC at commit b9cb96d (AIOMFAC-web 3.14)
# The only change to the Fortran sources: AIOMFAC_inout's hard-coded `calcviscosity = .true.` becomes a switch
# (Mod_bench.bench_visc), so that activities can be timed with and without the AIOMFAC-VISC branch.
set -e
SRCDIR=$(realpath "$1"); WORK=$(realpath -m "$2"); HERE=$(dirname "$(realpath "$0")")
mkdir -p "$WORK/off" "$WORK/nobc"
cp "$SRCDIR"/*.f90 "$WORK/"; rm -f "$WORK"/*__genmod.f90 "$WORK/Main_IO_driver.f90"
cp "$HERE/bench_main.f90" "$HERE/Mod_bench.f90" "$WORK/"
cp -r "$SRCDIR/../Auxiliary" "$WORK/" 2>/dev/null || true
cd "$WORK"
python3 - <<'PY'
s = open("AIOMFAC_inout.f90").read()
s = s.replace("use, intrinsic :: IEEE_ARITHMETIC", "use, intrinsic :: IEEE_ARITHMETIC\nuse Mod_bench, only : bench_visc", 1)
s = s.replace("calcviscosity = .true.", "calcviscosity = bench_visc", 1)
open("AIOMFAC_inout.f90", "w").write(s)
PY
grep -q "calcviscosity = bench_visc" AIOMFAC_inout.f90
SRC="Mod_kind_param.f90 Mod_bench.f90 ModStringFunctions.f90 ModSystemProp.f90 Mod_MINPACK.f90 ModSubgroupProp.f90 ModCompScaleConversion.f90 ModSRparam.f90 ModAIOMFACvar.f90 ModMRpart.f90 ModOScommands.f90 ModPureCompProp.f90 ModComponentNames.f90 ModNumericalTransformations.f90 Mod_InputOutput.f90 ModViscEyring.f90 ModPureViscosPar.f90 ModSRunifac.f90 SubModDefSystem.f90 ModCalcActCoeff.f90 ModZSRvisc.f90 SubModDissociationEquil.f90 ModFiniteDiffSens.f90 zerobracket_inwards.f90 brent.f90 AIOMFAC_inout.f90 bench_main.f90"
# flags of the repository's build_command_line.txt, and -O3 -march=native without bounds checking
(cd off && gfortran -o ../bench_official.out -O3 -ffree-line-length-none -fstack-protector-strong -fbounds-check $(for f in $SRC; do echo ../$f; done))
(cd nobc && gfortran -o ../bench_O3.out -O3 -march=native -ffree-line-length-none $(for f in $SRC; do echo ../$f; done))
echo "built $WORK/bench_official.out and $WORK/bench_O3.out"
