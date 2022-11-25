const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available

import Pkg; Pkg.add("HDF5")
Pkg.add("Printf");Pkg.add("Statistics");Pkg.add("LinearAlgebra")
using HDF5

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra

import ParallelStencil: INDICES
ix,iy=INDICES[1],INDICES[2]
macro sum_IBM_steny(A) esc(:(sum($A[$ix,(:)]))) end
macro sum_IBM_stenx(A,idx) esc(:(sum($A[$idx[($ix+1)]]))) end




@parallel function compute_timesteps!(dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, Mus::Data.Array, Vsc::Data.Number, Ptsc::Data.Number, min_dxy2::Data.Number, max_nxy::Data.Number)
    @all(dτVx) = Vsc*min_dxy2/@av_xi(Mus)/4.1
    @all(dτVy) = Vsc*min_dxy2/@av_yi(Mus)/4.1
    @all(dτPt) = Ptsc*4.1*@all(Mus)/max_nxy
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, dτPt::Data.Array, dx::Data.Number, dy::Data.Number,ρ::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @all(Pt) - @all(dτPt)*@all(∇V)./ρ
    return
end

@parallel function compute_τ!(∇V::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, Mus::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(τxx) = 2.0*@all(Mus)*(@d_xa(Vx)/dx - 1.0/3.0*@all(∇V))
    @all(τyy) = 2.0*@all(Mus)*(@d_ya(Vy)/dy - 1.0/3.0*@all(∇V))
    @all(τxy) = 2.0*@av(Mus)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx))
    return
end

@parallel function compute_dV!(Rx::Data.Array, Ry::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, Pt::Data.Array, Rogx::Data.Array, Rogy::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, dampX::Data.Number, dampY::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(Rx)    = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx + @av_xi(Rogx)
    @all(Ry)    = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @av_yi(Rogy)
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array)
    @inn(Vx) = @inn(Vx) + @all(dτVx)*@all(dVxdτ)
    @inn(Vy) = @inn(Vy) + @all(dτVy)*@all(dVydτ)
    return
end

@parallel_indices (ix,iy) function bc_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix,iy) function bc_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end


###################### IBM functions ########################
#~~~~~~~~~~~~~~~~ parallel ~~~~~~~~~~~~~~~~~~~~#

@parallel function take_av(Vx::Data.Array,Vy::Data.Array,Vxin::Data.Array,Vyin::Data.Array)
    @all(Vxin)=@av_xa(Vx)
    @all(Vyin)=@av_ya(Vy)
    return
end

@parallel function compute_weigted_u_euler2lag!(u::Data.Array,v::Data.Array,IBM_delta::Data.Array,IBM_fx::Data.Array,IBM_fy::Data.Array)
    @all(IBM_fx)=@all(u)*@all(IBM_delta)
    @all(IBM_fy)=@all(v)*@all(IBM_delta)
    return
end

@parallel function sum_IBM!(IBM_fx::Data.Array,IBM_fy::Data.Array,IBM_fxTemp::Data.Array,IBM_fyTemp::Data.Array,IBM_fxLag::Data.Array,IBM_fyLag::Data.Array,IBM_sum_idx::Array{Int64,1},IBM_sten::Int)
    @all(IBM_fxTemp)=@sum_IBM_steny(IBM_fx)
    @all(IBM_fyTemp)=@sum_IBM_steny(IBM_fy)
    return
end

@parallel function IBM_velocity_correction!(IBM_vx::Data.Array,IBM_vy::Data.Array,Vx::Data.Array,Vy::Data.Array)
    @inn(Vx)=@inn(Vx)+@inn(IBM_vx)
    @inn(Vy)=@inn(Vy)+@inn(IBM_vy)
    return
end

# need to be parallelized
function getU_euler2lag(IBM_idxx,IBM_idxy,Vxin,Vyin,Vx_euler2lag,Vy_euler2lag,IBM_sten)
    numLag=size(IBM_idxx,1)
    for i=1:numLag
        Vx_euler2lag[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vxin[IBM_idxx[i,:],IBM_idxy[i,:]]
        Vy_euler2lag[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vyin[IBM_idxx[i,:],IBM_idxy[i,:]]
    end
    return Vx_euler2lag, Vy_euler2lag 
end

function sumEuler2Lag(IBM_fxTemp,IBM_fyTemp,IBM_fxLag,IBM_fyLag,IBM_sten,numLag)
    for i=1:numLag
	IBM_fxLag[i] = sum(IBM_fxTemp[Int((i-1)*IBM_sten+1):1:Int(i*IBM_sten)])
	IBM_fyLag[i] = sum(IBM_fyTemp[Int((i-1)*IBM_sten+1):1:Int(i*IBM_sten)])
    end
    return IBM_fxLag,IBM_fyLag
end

function lag2euler(IBM_lagIdx,IBM_lagDelta,IBM_fxd,IBM_fyd,IBM_vx_correction,IBM_vy_correction,iter)
    IBM_fx_correction=zeros(size(IBM_vy_correction,1),size(IBM_vx_correction,2))
    IBM_fy_correction=copy(IBM_fx_correction)
    for i=1:size(IBM_lagIdx,1)    
	for j=1:size(IBM_lagIdx,2)
	    if length(IBM_lagIdx[i,j])>0
		    IBM_fx_correction[i,j]=sum(IBM_fxd[IBM_lagIdx[i,j]].*IBM_lagDelta[i,j])
	       	IBM_fy_correction[i,j]=sum(IBM_fyd[IBM_lagIdx[i,j]].*IBM_lagDelta[i,j])
	    end
	end
    end
    IBM_vx_correction[1,:]=(3*IBM_fx_correction[1,:]-IBM_fx_correction[2,:])./2
    IBM_vx_correction[end,:]=(3*IBM_fx_correction[end,:]-IBM_fx_correction[end-1,:])./2
    IBM_vx_correction[2:end-1,:]=(IBM_fx_correction[1:end-1,:]+IBM_fx_correction[2:end,:])./2
    IBM_vy_correction[:,1]=(3*IBM_fy_correction[:,1]-IBM_fy_correction[:,2])./2
    IBM_vy_correction[:,end]=(3*IBM_fy_correction[:,end]-IBM_fy_correction[:,end-1])./2
    IBM_vy_correction[:,2:end-1]=(IBM_fy_correction[:,1:end-1]+IBM_fy_correction[:,2:end])./2
    return IBM_vx_correction,IBM_vy_correction
end
#~~~~~~~~~~~~~~~~ initialization ~~~~~~~~~~~~~~~~~~~~#
function getObjShape(IBM_mu,IBM_sigma,IBM_amp,X,IBM_bumpNum,IBM_s,lx,eps,a1,k0)
    IBM_lagXTemp = X
    IBM_lagYTemp = eps*a1.*sin.(k0.*IBM_lagXTemp)
    IBM_lagYTemp[IBM_lagYTemp.<=0] .= 0;
    return IBM_lagXTemp,IBM_lagYTemp 
end

function getLagPoints(IBM_lagXTemp,IBM_lagYTemp,IBM_mu,IBM_sigma,IBM_amp,IBM_bumpNum,IBM_s,dx,dy,lx,eps,a1,k0)
    # compute total Lag points needed
    segX=IBM_lagXTemp[2:end]-IBM_lagXTemp[1:end-1]
    segY=IBM_lagYTemp[2:end]-IBM_lagYTemp[1:end-1]
    seg=sqrt.(segX.^2+segY.^2)
    segLength=sum(seg)
    numLag=floor.(segLength/min(dx,dy))
    # define Lag points 
    IBM_lagX=0:lx/numLag:lx
    IBM_lagY=eps*a1.*sin.(k0.*IBM_lagX)
    IBM_lagY[IBM_lagY.<=0] .= 0;
    return IBM_lagX,IBM_lagY
end

function getNormalTangentDir(IBM_lagX,IBM_lagY,beta1,eps,a1,k0)
    IBM_sin=eps*beta1*a1.*sin.(k0.*IBM_lagX)
    # IBM_sin[IBM_lagY.<=0] .= 1;
    IBM_cos=eps*beta1*a1.*cos.(k0.*IBM_lagX)
    # IBM_cos[IBM_lagY.<=0] .= 0;
    for i=1:length(IBM_lagX)
        print("IBM_sin: ",IBM_sin[i],", IBM_cos: ",IBM_cos[i],"\n")
    end
    return IBM_sin,IBM_cos
end

function getDeltaIdx(IBM_lagX,IBM_lagY,nx,ny,X,Y,dx,dy)
    IBM_deltaIdx=zeros(length(IBM_lagX),2)
    x_idx_start,y_idx_start=1,1
    # define the matrix location around each lag point
    for lagCount=1:length(IBM_lagX)
        x_idx_start,y_idx_start=x_idx_start-5,y_idx_start-5
        x_idx_start,y_idx_start=Int64((x_idx_start.+abs.(x_idx_start))./2+1),Int64((y_idx_start.+abs.(y_idx_start))./2+1) # lower bound
        x_idx_end,y_idx_end=max(x_idx_start+10,nx-1),max(y_idx_start+10,ny-1)
        lagX,lagY=IBM_lagX[lagCount]-dx,IBM_lagY[lagCount]-dy
        # get the x lower bonud index of the current lag point
        for xcount=x_idx_start:x_idx_end
            if (X[xcount]<=lagX && X[xcount+1] >= lagX)
	        IBM_deltaIdx[lagCount,1]=Int64(xcount)
	    end
        # periodic
        if (lagX<=X[1]-dx)
                IBM_deltaIdx[lagCount,1]=Int64(nx-1)
	    elseif(lagX<=X[1])
                IBM_deltaIdx[lagCount,1]=Int64(nx)
        end
        if (lagX>=X[nx]); IBM_deltaIdx[lagCount,1]=nx; end
        end

        # get the z lower bound index of the current lag point
        for ycount=y_idx_start:y_idx_end
	    if (Y[ycount]<=lagY && Y[ycount+1]>=lagY)
	        IBM_deltaIdx[lagCount,2]=Int64(ycount)
	    end
	    if (lagY<=Y[1]); IBM_deltaIdx[lagCount,2]=Int64(1); end
	    if (lagY>=Y[ny]); IBM_deltaIdx[lagCount,2]=Int64(ny); end
	end
    end
    return IBM_deltaIdx
end

function getDeltaMatrix(IBM_deltaIdx,IBM_lagX,IBM_lagY,IBM_sten,nx,ny,lx,ly,X,Y,dx,dy)
    numLag=length(IBM_lagX)
    IBM_deltaIdxx,IBM_deltaIdxy=zeros(Int64,numLag,IBM_sten),zeros(Int64,numLag,IBM_sten)
    IBM_deltaMat=zeros(IBM_sten,IBM_sten,numLag)
    IBM_delta,IBM_idxx,IBM_idxy=zeros(numLag*IBM_sten,IBM_sten),zeros(Int64,numLag*IBM_sten,IBM_sten),zeros(Int64,numLag*IBM_sten,IBM_sten)
    for lagCount=1:numLag
	    phi_x,phi_y=zeros(1,IBM_sten),zeros(1,IBM_sten)
        x_corner,y_corner=Int64(IBM_deltaIdx[lagCount,1]),Int64(IBM_deltaIdx[lagCount,2]) # top left corner
	    xlag,ylag=IBM_lagX[lagCount],IBM_lagY[lagCount]
        x_corner_end=Int64(mod(x_corner+3,nx)); if(x_corner_end==0);x_corner_end=Int64(nx); end
        y_corner_end=Int64(mod(y_corner+3,ny)); if(y_corner_end==0);y_corner_end=Int64(ny); end
        if (x_corner<x_corner_end)
    	    IBM_deltaIdxx[lagCount,:]= x_corner:x_corner_end
        else
     	    IBM_deltaIdxx[lagCount,:]=[x_corner:nx;1:x_corner_end]
        end
        if (y_corner<y_corner_end)
    	    IBM_deltaIdxy[lagCount,:]= y_corner:y_corner_end
        else
    	    IBM_deltaIdxy[lagCount,:]=[y_corner:ny;1:y_corner_end]
        end

        rx,ry=(X[x_corner:x_corner_end].-xlag)./dx,(Y[y_corner:y_corner_end].-ylag)./dy
        if (lagCount>numLag/2)
	    if(x_corner>x_corner_end);rx=([X[x_corner:nx];X[1:x_corner_end].+lx].-xlag)./dx;end
    	    if(y_corner>y_corner_end);ry=([Y[y_corner:ny];Y[1:y_corner_end].+ly].-ylag)./dy;end
        else
    	    if(x_corner>x_corner_end);rx=([X[x_corner-1:nx-1].-lx;X[1:x_corner_end]].-xlag)./dx;end
            if(y_corner>y_corner_end);ry=([Y[y_corner:ny].-lx;Y[1:y_corner_end]].-ylag)./dy;end
        end

	# get phi in each direction, 4 points stencil, see Peskin eq 6.27
	for stenCount=1:IBM_sten
	    rx_cur,ry_cur=rx[stenCount],ry[stenCount]
	    rx_cur_abs,ry_cur_abs=abs(rx_cur),abs(ry_cur)
	    # equ 6.27 Peskin 2002
	    # x
	    if (rx_cur_abs>=2)
		    phi_x[stenCount]=0
	    elseif (rx_cur_abs>=1 && rx_cur_abs<=2)
		    phi_x[stenCount]=(5-2*rx_cur_abs-sqrt(-7+12*rx_cur_abs-4*rx_cur^2))/8
	    elseif (rx_cur_abs<=1)
		    phi_x[stenCount]=(3-2*rx_cur_abs+sqrt(1+4*rx_cur_abs-4*rx_cur^2))/8
	    end
	    # y
	    if (ry_cur_abs>=2)
		    phi_y[stenCount]=0
	    elseif (ry_cur_abs>=1 && ry_cur_abs<=2)
		    phi_y[stenCount]=(5-2*ry_cur_abs-sqrt(-7+12*ry_cur_abs-4*ry_cur^2))/8
	    elseif (ry_cur_abs<=1)
        	phi_y[stenCount]=(3-2*ry_cur_abs+sqrt(1+4*ry_cur_abs-4*ry_cur^2))/8
        end
	end
	# ensure conservation
        phi_yn=copy(phi_y)
       
	if (abs(sum(phi_x'*phi_y)-1)>1e-4)
	    phi_yn[1]=phi_y[1]/sum(phi_x'*phi_y)
	    phi_yn[2]=phi_y[2]/sum(phi_x'*phi_y)
	    phi_yn[3]=phi_y[3]/sum(phi_x'*phi_y)
	end

	# combine x and y, eq 4 and eq 5, Kempe 2012
	IBM_deltaMat[:,:,lagCount]=phi_x'*phi_yn
	# reshape the delta matrix for parallel
	IBM_delta[(lagCount-1)*IBM_sten+1:lagCount*IBM_sten,:]=phi_x'*phi_yn
	IBM_idxx[(lagCount-1)*IBM_sten+1:lagCount*IBM_sten,:]=[IBM_deltaIdxx[lagCount,:] IBM_deltaIdxx[lagCount,:] IBM_deltaIdxx[lagCount,:] IBM_deltaIdxx[lagCount,:]]
	IBM_idxy[(lagCount-1)*IBM_sten+1:lagCount*IBM_sten,:]=[IBM_deltaIdxy[lagCount,:] IBM_deltaIdxy[lagCount,:] IBM_deltaIdxy[lagCount,:] IBM_deltaIdxy[lagCount,:]]'
    end 
    return IBM_deltaIdxx,IBM_deltaIdxy,IBM_delta
end


function getlagIdxforEulerloop(IBM_lagX,IBM_lagY,IBM_delta,nx)
    idxy_max=maximum(IBM_lagY)
    IBM_lagIdx=Array{Array{Int64,1}}(undef,nx,idxy_max)
    IBM_lagDelta=Array{Array{Float64,1}}(undef,nx,idxy_max)
    for i=1:nx, j=1:idxy_max
	IBM_lagIdx[i,j],IBM_lagDelta[i,j]=[],[]
    end
    for i=1:size(IBM_lagX,1)
	for j=1:size(IBM_lagX,2)
	    for k=1:size(IBM_lagY,2)
		push!(IBM_lagIdx[IBM_lagX[i,j],IBM_lagY[i,k]],i)
		push!(IBM_lagDelta[IBM_lagX[i,j],IBM_lagY[i,k]],IBM_delta[(i-1)*size(IBM_lagY,2)+j,k])
	    end
	end
    end
    return IBM_lagIdx,IBM_lagDelta
end

##################################################


##################################################
@views function Stokes2D()
    # Physics
    PI        = 3.14159265358793
    lx, ly    = 20*PI, 5*PI  # domain extends
    μs0       = 0.01          # matrix viscosity
    ρ	      = 1.0 	     # ice density
    G         = 0.0          # gravity
    ρg        = ρ*G	         
    alpha     = 0	         # tilt angle, in degrees
    epsi_bc   = 1
    # Numerics
    iterMax   = 1e5         # maximum number of pseudo-transient iterations
    nout      = 1000        # error checking frequency
    Vdmp      = 4.0         # damping paramter for the momentum equations
    Vsc       = 1.0         # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc      = 1.0         # relaxation paramter for the pressure equation pseudo-timestep limiter
    ε         = 1e-6        # nonlinear absolute tolerence
    ε_rel     = 1e-8
    nx, ny    = 127, 31     # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    # Derived numerics
    dx, dy    = lx/(nx), ly/(ny) # cell sizes
    min_dxy2  = min(dx,dy)^2
    max_nxy   = max(nx,ny)*1.0
    dampX     = 1.0-Vdmp/nx # damping term for the x-momentum equation
    dampY     = 1.0-Vdmp/ny # damping term for the y-momentum equation
    # Array allocations
    Pt        = @zeros(nx  ,ny  )
    dτPt      = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    Rx        = @zeros(nx-1,ny-2)
    Ry        = @zeros(nx-2,ny-1)
    dVxdτ     = @zeros(nx-1,ny-2)
    dVydτ     = @zeros(nx-2,ny-1)
    dτVx      = @zeros(nx-1,ny-2)
    dτVy      = @zeros(nx-2,ny-1)
    # data outputs   
    filepath= "./results/"
    filename= "nye_PerioBC_reso_"*string(nx)*"_"*string(ny)*".h5"
    # grid output
    print("grid size: ",nx,"x",ny,", dx:",dx,", dy:",dy,"\n")

    ########### IBM  ##########
    # ~~ parameters
    IBM_mu      = lx*2/6
    IBM_sigma   = 0.1
    IBM_amp     = 1
    IBM_sten    = Int(4)
    IBM_us      = 0
    IBM_lambda  = 50
    IBM_bumpNum = lx/IBM_lambda
    IBM_s       = 0
    phi2	    = 1
    idxx=1; idxy=256;
    ###########################


    # Initial conditions
    Radc      =  zeros(nx  ,ny  )
    Rogx      =  ρg*sind(alpha)*ones(nx,ny)
    Rogy      =  ρg*cosd(alpha)*ones(nx,ny)
    Mus       =  μs0*ones(nx,ny)
    Mus       = Data.Array(Mus)
    etas      = @zeros(nx-1,ny-1)
    Rogx      = Data.Array(Rogx)
    Rogy      = Data.Array(Rogy)

    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; 
    loadpath = "./viz2D_out/"; numericalp = Animation(loadpath,String[]); 
    errorp = Animation(loadpath,String[]); analyticalp = Animation(loadpath,String[]); 

    ########### IBM setup ##########
    # ~~ Nye analytical parameters
    k0      =    1.0
    eps     =    3.0
    a1      =    1.0
    amp     = 0.0
    X, Y    = -amp:dx:lx-dx-amp, -amp:dy:ly-dy-amp
    Xv, Yv  = dx-amp:dx:(lx+dx-amp), dy-amp:dy:(ly+dy-amp)
    X2, Y2  = Array(X)*ones(1,size(Y,1)), ones(size(X,1),1)*Array(Y)'
    Xv2, Yv2= Array(Xv)*ones(1,size(Yv,1)), ones(size(Xv,1),1)*Array(Yv)'
    # ~~ initialzation
    IBM_lagXTemp,IBM_lagYTemp=getObjShape(IBM_mu,IBM_sigma,IBM_amp,X,IBM_bumpNum,IBM_s,lx,eps,a1,k0)
    IBM_lagX,IBM_lagY=getLagPoints(IBM_lagXTemp,IBM_amp*IBM_lagYTemp,IBM_mu,IBM_sigma,IBM_amp,IBM_bumpNum,IBM_s,dx,dy,lx,eps,a1,k0)
    IBM_ud,IBM_wd=IBM_us,IBM_us 
    # ~~ compute delta functions
    IBM_deltaIdx=getDeltaIdx(IBM_lagX,IBM_lagY,nx,ny,X,Y,dx,dy)
    IBM_idxx,IBM_idxy,IBM_delta=getDeltaMatrix(IBM_deltaIdx,IBM_lagX,IBM_lagY,IBM_sten,nx,ny,lx,ly,X,Y,dx,dy)
    numLag=size(IBM_idxx,1)
    IBM_lagIdx,IBM_lagDelta=getlagIdxforEulerloop(IBM_idxx,IBM_idxy,IBM_delta,nx)
    IBM_sum_idx=collect(1:IBM_sten:size(IBM_delta,1))
    # ~~ allocation
    IBM_fx,IBM_fy=@zeros(nx+1,ny), @zeros(nx,ny+1)
    IBM_qxT,IBM_qyT=@zeros(nx+1,ny), @zeros(nx,ny+1)
    Vx_euler2lag=zeros(IBM_sten*numLag,IBM_sten)
    Vy_euler2lag=zeros(IBM_sten*numLag,IBM_sten)
    IBM_fx=@zeros(IBM_sten*numLag,IBM_sten)
    IBM_fy=@zeros(IBM_sten*numLag,IBM_sten)
    IBM_fxTemp=@zeros(IBM_sten*numLag,1)
    IBM_fyTemp=@zeros(IBM_sten*numLag,1)
    IBM_fxLag=@zeros(numLag,1)
    IBM_fyLag=@zeros(numLag,1)
    ud=@zeros(numLag,1)
    vd=@zeros(numLag,1)
    IBM_vx_correction,IBM_vy_correction=@zeros(nx+1,ny),@zeros(nx,ny+1)
    Vxin,Vyin=@zeros(nx,ny),@zeros(nx,ny)
    Rxin,Ryin=@zeros(nx,ny),@zeros(nx,ny)
    include_IBM=1
    ###########################


    ########### analytical soln  ##########
    # ~~ parameters
    U       =    1
    eta_reg =    0.005
    k_reg   =    0.8
    beta1   =    k_reg^2/(k_reg^2+k0^2)
    u1      =    zeros(Float64,nx,ny)
    v1,Vxa,Vya,Pa = copy(u1), copy(u1), copy(u1), copy(u1)
    amp = 0.0;
    for ziter=1:ny
        for xiter=1:nx
	    u1[xiter,ziter]  = eps*U*Y[ziter]*a1*beta1*k0^2*exp(-k0*(Y[ziter]-amp))*sin(k0*X[xiter])
        Vxa[xiter,ziter] = u1[xiter,ziter]+U
	    v1[xiter,ziter]  = eps*U*a1*beta1*k0*(1+k0*(Y[ziter]-amp))*exp(-k0*(Y[ziter]-amp))*cos(k0*X[xiter])
        Vya[xiter,ziter] = v1[xiter,ziter]
        Pa[xiter,ziter]  = 2*eps*eta_reg*U*a1*beta1*k0^2*exp(-k0*(Y[ziter]-amp))*cos(k0*X[xiter])
       end
    end
    Vx_lb, Vx_rb  = zeros(1,ny), zeros(1,ny); Vx_tb, Vx_bb  = zeros(nx+1,1), zeros(nx+1,1)
    Vy_lb, Vy_rb  = zeros(1,ny+1), zeros(1,ny+1); Vy_tb, Vy_bb  = zeros(nx,1), zeros(nx,1)

    Vx_lb .= (3*Vxa[1,:]'.-Vxa[2,:]')./2; Vx_rb .= (3*Vxa[end,:]'.-Vxa[end-1,:]')./2
    Vx_tb[2:end-1] .= (Vxa[1:end-1,end].+Vxa[2:end,end])./2; Vx_bb[2:end-1] .= (Vxa[1:end-1,1].+Vxa[2:end,1])./2
    Vx_tb[1] = Vx_lb[end]; Vx_tb[end] = Vx_rb[end]; Vx_bb[1] = Vx_lb[1]; Vx_bb[end] = Vx_rb[1]

    Vy_bb .= (3*Vya[:,1].-Vya[:,2])./2; Vy_tb .= (3*Vya[:,end].-Vya[:,end-1])./2
    Vy_lb[2:end-1] .= (Vya[1,1:end-1].+Vya[1,2:end])./2; Vy_rb[2:end-1] .= (Vya[end,1:end-1].+Vya[end,2:end])./2
    Vy_lb[1] = (2*Vya[1,1]-Vy_lb[2]); Vy_lb[end] = (2*Vya[1,end]-Vy_lb[end-1])
    Vy_rb[1] = (2*Vya[end,1]-Vy_rb[2]); Vy_rb[end] =  (2*Vya[end,end]-Vy_rb[end-1])
    H=IBM_lagYTemp*ones(1,size(X2,2));
    Vx_euler2lag,Vy_euler2lag=getU_euler2lag(IBM_idxx,IBM_idxy,Vxa,Vya,Vx_euler2lag,Vy_euler2lag,IBM_sten)
    @parallel compute_weigted_u_euler2lag!(Vx_euler2lag,Vy_euler2lag,IBM_delta,IBM_fx,IBM_fy)
    @parallel sum_IBM!(IBM_fx,IBM_fy,IBM_fxTemp,IBM_fyTemp,IBM_fxLag,IBM_fyLag,IBM_sum_idx,IBM_sten)
    ud,vd=sumEuler2Lag(IBM_fxTemp,IBM_fyTemp,ud,vd,IBM_sten,numLag)	
    ###########################



    # Time loop
    @parallel compute_timesteps!(dτVx, dτVy, dτPt, Mus, Vsc, Ptsc, min_dxy2, max_nxy)
    err=2*ε; err_old=20000*ε; err_rel = abs(err-err_old)
    iter=1; niter=0; err_evo1=[]; err_evo2=[]    
    while (err > ε && err_rel>ε_rel) && iter <= iterMax
        if (iter==1)  global wtime0 = Base.time()  end
        err_old = copy(err)
        @parallel compute_P!(∇V, Pt, Vx, Vy, dτPt, dx, dy, ρ)
        @parallel compute_τ!(∇V, τxx, τyy, τxy, Vx, Vy, Mus, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVxdτ, dVydτ, Pt, Rogx, Rogy, τxx, τyy, τxy, dampX, dampY, dx, dy)
        @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
        @parallel take_av(Vx,Vy,Vxin,Vyin)
        # ~~~~~~~~~~~~~~~ IBM ~~~~~~~~~~~~~~~~~~~ #
	if (include_IBM==1)
        @parallel take_av(Vx,Vy,Vxin,Vyin)
        Vx_euler2lag,Vy_euler2lag=getU_euler2lag(IBM_idxx,IBM_idxy,Vxin,Vyin,Vx_euler2lag,Vy_euler2lag,IBM_sten)
        @parallel compute_weigted_u_euler2lag!(Vx_euler2lag,Vy_euler2lag,IBM_delta,IBM_fx,IBM_fy)
        @parallel sum_IBM!(IBM_fx,IBM_fy,IBM_fxTemp,IBM_fyTemp,IBM_fxLag,IBM_fyLag,IBM_sum_idx,IBM_sten)
        IBM_fxLag,IBM_fyLag=sumEuler2Lag(IBM_fxTemp,IBM_fyTemp,IBM_fxLag,IBM_fyLag,IBM_sten,numLag)	
        IBM_fxd,IBM_fyd=(ud.-IBM_fxLag),(vd.-IBM_fyLag)
	    IBM_vx_correction,IBM_vy_correction=lag2euler(IBM_lagIdx,IBM_lagDelta,IBM_fxd,IBM_fyd,IBM_vx_correction,IBM_vy_correction,iter)
        @parallel IBM_velocity_correction!(IBM_vx_correction,IBM_vy_correction,Vx,Vy)
        @parallel take_av(Vx,Vy,Vxin,Vyin)
	end	
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


        # boundary conditions
        Vx[:,1] .= Vx_bb; Vy[:,1]=Vy_bb; Vx[:,end]= Vx_tb; Vy[:,end].=Vy_tb
        Pt[:,1] .= Pa[:,1]; Pt[:,end] .= Pa[:,end]; Pt[1,:].= 1*Pa[1,:]; Pt[end,:] .= 1*Pa[end,:]
        Vx[end,:] .= Vx_rb'; Vy[end,:] .= 1*Vy_rb'; Vx[1,:] .= Vx_rb'; Vy[1,:] .= 1*Vy_lb'
	
        global mean_Rx, mean_Ry, mean_∇V
   	    Rxin[2:end-1,2:end-1]=(Rx[1:end-1,:].+Rx[2:end,:])/2
  	    Ryin[2:end-1,2:end-1]=(Ry[:,1:end-1].+Ry[:,2:end])/2
        Vxa[Y2.<=H].=0;Vya[Y2.<=H].=0;Pa[Y2.<=H].=0;
        Vxin[Y2.<=H].=0;Vyin[Y2.<=H].=0;Pt[Y2.<=H].=0;
        norm2_vx, norm2_vy, norm2_p = sqrt(sum((Vxa.-Vxin).^2*dx*dy)), sqrt(sum((Vya.-Vyin).^2*dx*dy)), sqrt(sum((Pa.-Pt).^2)*dx*dy)
        mean_Rx = sqrt(sum(Rxin.^2)); mean_Ry = sqrt(sum(Ryin.^2)); mean_∇V = sqrt(sum(∇V.^2))
        err = maximum([norm2_vx, norm2_vy, norm2_p])
        err_rel = abs(err-err_old);
        err_old=err;
        push!(err_evo1, err); push!(err_evo2,iter)
       if mod(iter,nout)==0
        push!(err_evo1, err); push!(err_evo2,iter)
        @printf("Total steps = %d, err = %1.3e, err_rel = %1.3e [norm_Vx=%1.3e, norm_Vy=%1.3e, norm_P=%1.3e] \n", iter, err, err_rel, norm2_vx, norm2_vy, norm2_p)     
        end
        iter+=1; niter+=1
    end



    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(niter-10)                     # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
    @printf("Total steps = %d, err = %2.3e, time = %1.3e min (@ T_eff = %1.2f GB/s) \n", niter, err, wtime/60, round(T_eff, sigdigits=2))

    A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(niter-10)                     # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
    @printf("Total steps = %d, err = %2.3e, time = %1.3e min (@ T_eff = %1.2f GB/s) \n", niter, err, wtime/60, round(T_eff, sigdigits=2))

    # Visualisation
    Vxa[Y2.<=H].=0;Vya[Y2.<=H].=0;Pa[Y2.<=H].=0;
    Vxin[Y2.<=H].=0;Vyin[Y2.<=H].=0;Pt[Y2.<=H].=0;
    p1 = heatmap(X, Y, Array(Pt)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:inferno, title="Pressure, numerical")
    p2 = heatmap(X, Y, Array(Vxin)', aspect_ratio=1, xlims=(Xv[1],Xv[end]), ylims=(Y[1],Y[end]), c=:inferno, title="Vx, numerical")
    p3 = heatmap(X, Y, Array(Vyin)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:inferno, title="Vy, numerical")
    p4 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="error", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
    p5 = heatmap(X, Y, Array(Pa)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:inferno, title="Pressure, analytical")
    p6 = heatmap(X, Y, Array(Vxa)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:inferno, title="Vx, analytical")
    p7 = heatmap(X, Y, Array(Vya)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:inferno, title="Vy, analytical")
    # error plot    
    p8 = heatmap(X,Y,Array(Vxin-Vxa)',aspect_ratio=1,xlims=(Xv[1],X[end]),ylims=(Y[1],Y[end]),c=:inferno,title="Vx-Vxa")
    p9 = heatmap(X,Y,Array(Vyin-Vya)',aspect_ratio=1,xlims=(X[1],X[end]),ylims=(Y[1],Y[end]),c=:inferno,title="Vy-Vya")
    p10 = heatmap(X,Y,Array(Pt-Pa)',aspect_ratio=1,xlims=(X[1],X[end]),ylims=(Y[1],Y[end]),c=:inferno,title="P-Pa")

    # display(plot(p1, p2, p4, p5))
    plot(p1, p2, p3, p4);
    png("Stokes2D_nye_numerical")
    plot(p5, p6, p7);
    png("Stokes2D_nye_analytical")
    plot(p8, p9, p10);
    png("Stokes2D_nye_err")
    norm2_vx, norm2_vy, norm2_p = sqrt(sum((Vxa.-Vxin).^2*dx*dy)), sqrt(sum((Vya.-Vyin).^2*dx*dy)), sqrt(sum((Pa.-Pt).^2)*dx*dy)
    
    fid=h5open(filepath*filename,"w")
    fid["Vx"  ] = Vx
    fid["Vy"  ] = Vy
    fid["Vxin"] = Vxin
    fid["Vyin"] = Vyin
    fid["Pt"  ] = Pt
    fid["Vya" ] = Vya
    fid["Vxa" ] = Vxa
    fid["Pa"  ] = Pa
    fid["X2"  ] = X2
    fid["Y2"  ] = Y2
    close(fid)

    print("\n ERROR relative to analytical soln:\n  Vx:", norm2_vx,", Vy:",norm2_vy,", P:",norm2_p,"\n")

    return
end

Stokes2D()