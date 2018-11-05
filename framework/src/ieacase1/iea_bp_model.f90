! Created by Jared J. Thomas, 2018
! FLight Optimization and Wind Laboratory (FLOW Lab)
! Brigham Young University

! implementation of the Bastankhah and Porte Agel (BPA) wake model for IEA case studies case 1
subroutine iea_bp_model_fortran(nTurbines, turbineXw, turbineYw, rotorDiameter, wind_speed, wec, wtVelocity)

    ! independent variables: turbineXw turbineYw rotorDiameter

    ! dependent variables: wtVelocity

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw
    real(dp), dimension(nTurbines), intent(in) :: rotorDiameter
    real(dp), intent(in) :: wind_speed, wec


    ! local (General)
    real(dp), dimension(nTurbines) :: loss, loss_array
    real(dp) :: CT, k, x, y, sigma, exponent, radical
    Integer :: turb, turbI

    ! model out
    real(dp), dimension(nTurbines), intent(out) :: wtVelocity

    intrinsic sqrt, exp, sum

    !"""Return each turbine's total loss due to wake from upstream turbines"""
    ! Equations and values explained in <iea37-wakemodel.pdf>

    ! Constant thrust coefficient
    CT = 4.0_dp*1.0_dp/3.0_dp*(1.0_dp-1.0_dp/3.0_dp)

    ! Constant, relating to a turbulence intensity of 0.075
    k = 0.0324555_dp

    ! Array holding the wake deficit seen at each turbine
    loss = 0.0_dp
    wtVelocity = 0.0_dp

    do, turbI=1, nTurbines             ! Looking at each turb (Primary)
        loss_array = 0.0_dp
        do, turb=1, nTurbines        ! Looking at all other turbs (Target)
            x = turbineXw(turbI) - turbineXw(turb)   ! Calculate the x-dist
            y = turbineYw(turbI) - turbineYw(turb)   ! And the y-offset
            if (x > 0.0_dp) then                   ! If Primary is downwind of the Target
                sigma = k*x + rotorDiameter(turb)/sqrt(8.0_dp)  ! Calculate the wake loss
                sigma = k*x + rotorDiameter(turb)/sqrt(8.0_dp)  ! Calculate the wake loss
                ! Simplified Bastankhah Gaussian wake model
                exponent = -0.5_dp * (y/(wec*sigma))**2

                radical = 1.0_dp - CT/(8.0_dp*sigma**2 / rotorDiameter(turb)**2)
                loss_array(turb) = (1.0_dp-sqrt(radical)) * exp(exponent)
            end if
            ! Note that if the Target is upstream, loss is defaulted to zero
            loss(turbI) = loss(turbI) + loss_array(turb)**2
        end do
        ! Total wake losses from all upstream turbs, using sqrt of sum of sqrs

        loss(turbI) = sqrt(loss(turbI))

        ! Effective windspeed is freestream multiplied by wake deficits
        wtVelocity(turbI) = wind_speed*(1.-loss(turbI))
    end do

end subroutine iea_bp_model_fortran
