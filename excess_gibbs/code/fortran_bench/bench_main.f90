! Timing driver for the Fortran AIOMFAC (andizuend/AIOMFAC, AIOMFAC-web v3.14) -- one CPU thread.
! Input (list-directed, from file given as argument 1):
!   nsys
!   per system:  id  ncp  npts  nrep_build  nrep_eval
!                per component: nsub  sg_1 q_1 ... sg_nsub q_nsub
!                per point:     T_K  w_2 ... w_ncp           (mass fractions of components 2..ncp; water = rest)
! Output (stdout), per system:  id  ncp  build_us  eval_us(no visc, median over points)  eval_visc_us  errflag
program bench_main
use Mod_kind_param, only : wp
use ModSystemProp, only : errorflag_clist, errorflagmix, nindcomp, NKNpNGS, SetSystem, topsubno, maxsmileslength
use ModSubgroupProp, only : SubgroupAtoms, SubgroupNames
use ModMRpart, only : MRdata
use ModSRparam, only : SRdata
use ModComponentNames, only : nametab
use Mod_bench, only : bench_visc
use Mod_InputOutput, only : cpsmiles
implicit none
integer, parameter :: maxc = 60, maxp = 50
character(len=3000) :: fin
character(len=maxsmileslength+7), dimension(maxc) :: cpname
character(len=maxsmileslength+7), dimension(:), allocatable :: outnames
integer :: nsys, s, id, ncp, npts, nrb, nre, dovisc, c, nsub, k, sg, q, p, r, nspecies, warningflag, iflag
integer, dimension(maxc, 0:topsubno) :: cpsubg0
integer, dimension(:,:), allocatable :: cpsubg
real(wp) :: T(maxp), W(maxp, maxc), t_b, t_e(maxp), t_v(maxp)
real(wp), dimension(:), allocatable :: inputconc, outputviscvars
real(wp), dimension(:,:), allocatable :: outputvars
logical, dimension(size(errorflag_clist)) :: errflag
integer(8) :: c0, c1, rate

call get_command_argument(1, fin)
call nametab()
call MRdata(); call SRdata(); call SubgroupNames(); call SubgroupAtoms()
call system_clock(count_rate=rate)
open(11, file=trim(fin), status='old', action='read')
read(11,*) nsys
allocate(cpsubg(maxc, topsubno))
do s = 1, nsys
    read(11,*) id, ncp, npts, nrb, nre, dovisc
    cpsubg = 0
    do c = 1, ncp
        read(11,*) nsub, (cpsubg0(c, 2*k-1), cpsubg0(c, 2*k), k = 1, nsub)
        do k = 1, nsub
            sg = cpsubg0(c, 2*k-1); q = cpsubg0(c, 2*k)
            cpsubg(c, sg) = q
        enddo
        write(cpname(c), '(A,I0)') 'bench_cp', c
    enddo
    do p = 1, npts
        read(11,*) T(p), W(p, 2:ncp)
        W(p, 1) = 1.0_wp - sum(W(p, 2:ncp))
    enddo
    ! model construction (SetSystem), repeated
    call system_clock(c0)
    do r = 1, nrb
        if (allocated(cpsmiles)) deallocate(cpsmiles)
        allocate(cpsmiles(ncp)); cpsmiles = ""
        call SetSystem(1, .true., ncp, cpname(1:ncp), cpsubg(1:ncp, 1:topsubno))
    enddo
    call system_clock(c1)
    t_b = real(c1 - c0, wp) / real(rate, wp) / nrb
    iflag = errorflagmix
    if (allocated(inputconc)) deallocate(inputconc, outputvars, outputviscvars, outnames)
    allocate(inputconc(nindcomp), outputvars(6, NKNpNGS), outputviscvars(2), outnames(NKNpNGS))
    do p = 1, npts
        inputconc = 0.0_wp
        inputconc(1:ncp) = W(p, 1:ncp)
        bench_visc = .false.
        call AIOMFAC_inout(inputconc, .false., T(p), nspecies, outputvars, outputviscvars, outnames, errflag, warningflag)
        call system_clock(c0)
        do r = 1, nre
            call AIOMFAC_inout(inputconc, .false., T(p), nspecies, outputvars, outputviscvars, outnames, errflag, warningflag)
        enddo
        call system_clock(c1)
        t_e(p) = real(c1 - c0, wp) / real(rate, wp) / nre
        if (any(errflag)) iflag = iflag + 1000 * findloc(errflag, .true., dim=1)
        t_v(p) = -1.0e-6_wp
        if (dovisc == 0) cycle
        bench_visc = .true.
        call system_clock(c0)
        do r = 1, max(1, nre / 10)
            call AIOMFAC_inout(inputconc, .false., T(p), nspecies, outputvars, outputviscvars, outnames, errflag, warningflag)
        enddo
        call system_clock(c1)
        t_v(p) = real(c1 - c0, wp) / real(rate, wp) / max(1, nre / 10)
    enddo
    write(*, '(I6, I4, 3ES14.5, I8, ES24.15)') id, ncp, 1.0e6_wp * t_b, 1.0e6_wp * median(t_e(1:npts)), &
        & 1.0e6_wp * median(t_v(1:npts)), iflag, outputvars(5, 1)
enddo
close(11)

contains
    real(wp) function median(a)
    real(wp), intent(in) :: a(:)
    real(wp) :: b(size(a)), tmp
    integer :: i, j, n
    b = a; n = size(b)
    do i = 2, n
        tmp = b(i); j = i - 1
        do while (j >= 1)
            if (b(j) <= tmp) exit
            b(j+1) = b(j); j = j - 1
        enddo
        b(j+1) = tmp
    enddo
    if (mod(n, 2) == 1) then
        median = b((n+1)/2)
    else
        median = 0.5_wp * (b(n/2) + b(n/2+1))
    endif
    end function median
end program bench_main
