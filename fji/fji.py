import numpy as np
# photutils 1.0.1




class VelSpace():
    def __init__(self,q, K2):
        if q <= 0:
            return("ERROR: q = m2/m1 = " +str(q) + " <= 0")
        
        self.q = q
        self.K2 = K2
    
    # =============================================================================
    # Funciones auxilares         
    # =============================================================================
    
    def xl1(self):        
        """ Returns the distance of the L1 point from star 1 scaled by the orbital separation.
        Param q mass ratio = m2/m1.
        """
        import numpy as np
        q = self.q
        
     
        NMAX   = 1000 # Maximun of iteration
        EPS = 1e-12 #
        
        # Coefficients definition:
        mu = q/(1+q)
        
        a1 = -1+mu; a2 =  2-2*mu; a3 = -1+mu; a4 =  1+2*mu; a5 = -2-mu; a6 =  1
    
        d1 = 1*a2; d2 = 2*a3; d3 = 3*a4; d4 = 4*a5; d5 = 5*a6
    
        # Iteration
        n = 0
        xold = 0
        x    = 1/(1+ q)
        
        while n < NMAX and np.abs(x-xold) > EPS*np.abs(x):
            xold = x
            f    = x*(x*(x*(x*(x*a6+a5)+a4)+a3)+a2)+a1
            df   = x*(x*(x*(x*d5+d4)+d3)+d2)+d1
            x   -= f/df
            n+=1
            
            if(n == NMAX):
                return("Error: exceeded maximum iterations")
        
        return x
    
    def rpot(self,x,y,z):
        import numpy as np
        q = self.q
        
        """ Computes the Roche potential at a given point. T
        his is for the standard synchronised Roche geometry
        q mass ratio = M2/M1 
        x,y,z the the poit coord in units scaled by separation
        """
        mu   = q/(1+q)
        comp = 1-mu
        x2y2 = x**2 + y**2
        z2   = z**2
        r1sq = x2y2+z2    
        r1   = np.sqrt(r1sq)
        r2   = np.sqrt(r1sq + 1 - 2*x)
        
        return (-comp/r1-mu/r2-(x2y2+mu*(mu-2*x))/2)
      

    #==============================================================================
    # Roche Lobe
    #==============================================================================
    
    def lobe(self,lobe_n, n=100):
        """ lobe2 returns arrays x and y for plotting an equatorial
        section of the Roche lobe of the secondary star in a binary of mass
        ratio q = M2/M1.  The arrays start and end at the inner lagrangian
        point and march around uniformly in azimuth looking from the centre of
        mass of the primary star.  n is the number of points and must be at
        least 3.
        Parama:

        n number of x and y values
        
        """
        
        import numpy as np
     
        xl1 = self.xl1
        rpot = np.vectorize(self.rpot)
     
        
        if n < 3:
            return("ERROR: n = " +str(n) + " < 3")

        # Esta constante se usa mas adelante cuando se buscan las raices: minimum accuracy in returned root
        FRAC = 1e-6
        
        # Compute L1 point and critical potential there.
        
        rl1 = xl1()
        cpot  = rpot(rl1,0,0)
        
        # Indice que varia segun el lobulo que estemos intando
        l={}; l[1]=[1,rl1,0]; l[2]=[-1,1-rl1,1]
            
        
      # Now compute Roche lobe in n regular steps of angle looking from centre of Roche lobe (i.e. L1)
        x=np.zeros(n)
        y=np.zeros(n)
    
        for i in range(n):
            # L1 point is a special case because derivative becomes zero there. lambda is set so that after i=0, there is a decent starting multiplier.
            if i ==0 or i==n-1:
                x[i]   = rl1
                y[i]   = 0
            else:
                theta = 2*np.pi*i/(n-1)
                dx =  l[lobe_n][0] * np.cos(theta)
                dy =  np.sin(theta)
                
                upper = l[lobe_n][1]
                lower = upper/4
                
                steps=101
                res=1
                while res > FRAC:
                    r=np.linspace(lower,upper,steps)
                    x0 = l[lobe_n][2] + r * dx
                    y0 = r*dy
                    rpot0=rpot(x0,y0,0)
                    res=np.abs(rpot0-cpot).min()
                    index=np.abs(rpot0-cpot).argmin()
                    upper=r[index+1]
                    lower=r[index-1]    
                
                x[i] = l[lobe_n][2] + r[index]*dx
                y[i] = r[index] * dy
                
        return x,y
    
        
    def vlobe(self,lobe_n, n=100):
        """vlob2 computes secondary's Roche lobe in velocity coordinates returns arrays vx and vy for plotting an equatorial section
        of the Roche lobe of the secondary star in a binary of mass ratio q = M2/M1
        in Doppler coordinates. The arrays start and end at the inner Lagrangian 
        point and march around uniformly in azimuth looking from the centre of 
        mass of the primary star. n is the number of points and must be at least 3"""
        
        q = self.q
        K2 = self.K2
        lobe = self.lobe
        
        # Call lobe2 then transform appropriately
        x,y=lobe(lobe_n,n)
        
        mu = q/(1+q)
        
        tvx = K2*(q+1)*(- y)
        tvy = K2*(q+1)*(x - mu)
        
        return tvx,tvy
        
    
    #==============================================================================
    # Gas stream     
    #==============================================================================
    
    def strinit(self):
        """ strinit sets a particle just inside the L1 point with the correct velocity as given in Lubow and Shu.
        Params:
        q mass ratio = M2/M1
        r start position returned
        v start velocity returned """
        import numpy as np
        q = self.q
    
        SMALL = 1.e-5
        rl1 = self.xl1()
        mu = q/(1+q)
        a = (1-mu)/rl1**3+mu/(1-rl1)**3
        lambda1 = np.sqrt(((a-2) + np.sqrt(a*(9*a-8)))/2)
        m1  = (lambda1*lambda1-2*a-1)/2/lambda1
        
        r = np.array([rl1-SMALL,-m1*SMALL,0])
        v = np.array([-lambda1*SMALL,-lambda1*m1*SMALL,0])
        return r, v
        
    
    def vtrans(self, t_type, x, y, vx, vy):
        """ vtrans computes two velocity transforms, (1) a straight transform from rotating to inertial frame and 
        (2) an inertial frame velocity in the disc."""
        import numpy as np
        # type: 1 for rotating - > inertial, 2 for position to disc, 3 for rotating.
        # x: x position (units of separation)
        # y: y position (units of separation)
        # vx: x velocity (omega*a = 1 units)
        # vy: y velocity (omega*a = 1 units)

        #
        # When translating to inertial, the accretor velocity is added. If you want the velocity relative to this you must add mu = q/(1+q) to tvy before using it.
        q = self.q
        mu = q/(1+q)
        
        if t_type == 1:
            tvx = vx - y
            tvy = vy + x - mu
        elif t_type == 2:
            rad  = np.sqrt(x*x + y*y)
            vkep = 1/np.sqrt((1+q)*rad)
            tvx = -vkep*y/rad
            tvy =  vkep*x/rad-mu
        elif t_type == 3:
            tvx = vx
            tvy = vy
        else:
            print("Error in vtrans: did not recognize type = " +str(t_type) + ". Only 1, 2, or 3 supported.");
        
        return tvx, tvy
    
        
    def rocacc(self,rx,ry,rz,vx,vy,vz):
        """ rocacc calculates and returns the acceleration (in the rotating frame) in a Roche potential of a particle of given position and velocity.
        Params:
            q mass ratio = M2/M1
            r position, scaled in units of separation.
            v velocity, scaled in units of separation. """
        import numpy as np
        q = self.q

        f1 = 1/(1+q)
        f2 = f1*q
        yzsq = np.square(ry) + np.square(rz)
        r1sq = np.square(rx) + yzsq
        r2sq = np.square(rx-1) + yzsq
        fm1  = f1/(r1sq*np.sqrt(r1sq))
        fm2  = f2/(r2sq*np.sqrt(r2sq))
        fm3  = fm1+fm2;
        
        tmpx = -fm3*rx + fm2 + 2*vy + rx - f2
        tmpy = -fm3*ry - 2*vx + ry
        tmpz = -fm3*rz
        
        return np.array([tmpx,tmpy,tmpz])
    
    
    def vstream(self,step,n,t_type=1):
        # Vector donse se almacenaran las velocidades calculadas
        import numpy as np

        strinit = self.strinit
        rocacc = self.rocacc
        vtrans = self.vtrans
        
        q = self.q
        K2 = self.K2
    
        x = np.zeros(n)
        y = np.zeros(n)
        vx = np.zeros(n)
        vy = np.zeros(n)
        
        
        #==============================================================================
        # 1er punto: L1
        #==============================================================================
        # Posicion de L1 con velocidad inicial 0
        x[0] = self.xl1()
        y[0] =  0
        vx[0] = 0
        vy[0] = 0
        
        # vx[0],vy[0] = vtrans(q,1, x[0], y[0], 0, 0)
        
        #==============================================================================
        # 2do punto dentro de L1 (Lubow and Shu)    
        #==============================================================================
        x[1] = strinit()[0][0]
        y[1] = strinit()[0][1] 
        vx[1] = strinit()[1][0]
        vy[1] = strinit()[1][1]
        
        #==============================================================================
        # Bucle para calcular la aceleracion en cada paso     
        #==============================================================================
        s = np.arange(n-2)+1
        
        for i in s:
            # Aceleracion
            a = rocacc(x[i], y[i], 0, vx[i], vy[i], 0)
            
            # Posiciones
            x[i+1] = x[i]+ vx[i]*step + 0.5*a[0]*np.square(step)
            y[i+1] = y[i]+ vy[i]*step + 0.5*a[1]*np.square(step)
            # Velocidades
            vx[i+1]  = vx[i] + a[0]*step
            vy[i+1]  = vy[i] + a[1]*step
            
        return K2*(q+1)*np.float_(vtrans(t_type,x,y,vx,vy))
        





# =============================================================================
# Algoritmos de centrado 
# =============================================================================
def Centroid(data,coords,size=7,method='1dg',mask=None):
    """
    Entrada: imagen, lista de coordenadas. Busca el centroide en una region de radio "r" entorno a la posicion dada en "coords".
    Salida: array con las posisiones ajustadas y distancia entre las posisiones "d"

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
        
    coords : numpy array
        n coordinates in a (n,2) numpy array.
    
    size : number, optional
        Size in pixels of the section containing the start where perform the centroid. The default is 3.
    
    method : photutils.centroid
        com: Calculates the object “center of mass” from 2D image moments.
        quadratic: Calculates the centroid by fitting a 2D quadratic polynomial to the data.
        1dg: Calculates the centroid by fitting 1D Gaussians to the marginal x and y distributions of the data.
        2dg: Calculates the centroid by fitting a 2D Gaussian to the 2D distribution of the data.

    Returns
    -------
    None.

    """    
    
    # =============================================================================
    # Paquetes utilizados     
    # =============================================================================
    from photutils.centroids import centroid_com, centroid_quadratic,centroid_1dg, centroid_2dg
    from astropy.nddata.utils import Cutout2D
    from astropy.stats import sigma_clipped_stats
    
    # Diccionario con los distintos metodos de centrados     
    cent = {'com': centroid_com, 'quadratic': centroid_quadratic, '1dg': centroid_1dg, '2dg':centroid_2dg}
    
    # Vamos a definir una seccion de los datos
    cut = Cutout2D(data, coords, size=size)
    sec = cut.data
    
    #Calculamos el cielo solo dentro de la dregion selectionada y lo restamos        
    median = sigma_clipped_stats(sec, sigma=3.0)[1]
    sec = sec - median
    
    x_s, y_s = cent[method](sec, mask=mask)
    
    fit_coords = cut.to_original_position([x_s, y_s])
    
    
    return fit_coords

# =============================================================================
# Seleccion manual de coordenadas
# =============================================================================
def InteractiveCentroid(data,n_points, size=7, method='1dg', mask=None, cmap='Greys',p_min=5,p_max=95):
    """ Definicion
    """
    import numpy as np
    import tkinter
    import matplotlib
    matplotlib.use('TKAgg')
    import matplotlib.pyplot as plt
    from tkinter import messagebox #tkinter.TkVersion 8.6 

    # limites del contraste 
    vmin = np.percentile(data,p_min)
    vmax = np.percentile(data,p_max)
    
    # Equiquetas para las estrellas
    label=['Obj','Comp1','Comp2','Comp3','Comp4','Comp5','Comp6','Comp7','Comp8','Comp9']

    
    plt.close('all')    
    plt.ion(),plt.show()
    plt.figure(1,figsize=(7,7))

    happy = False
    centroid = []
    while happy == False:
        plt.cla()
        plt.title('Select Object first then '+str(n_points-1)+' comparison stars')
        plt.imshow(data,origin='lower',cmap=cmap, aspect='equal',vmin=vmin,vmax=vmax)
        plt.tight_layout()
        plt.show()


        # Coger puntos de manera interactiva
        point = plt.ginput(n=n_points,timeout=0,show_clicks=True,)
        
        coords = point

        # =============================================================================
        # Centrado
        # =============================================================================
        for coord in coords:
            ind = coords.index(coord)
            
            cent = Centroid(data,coord,size=size,method=method,mask=mask)
            centroid.append(cent)
            
            # Posiciones del cursor
           # plt.scatter(coord[0],coord[1],marker='+',color='b',zorder=2,alpha=0.3)
            # Posiciones ajustadas
            plt.scatter(cent[0],cent[1],marker='+',color='C3')
            plt.scatter(cent[0],cent[1], s=80, facecolors='none', edgecolors='C3',zorder=1)

            plt.annotate(label[ind],coord, xytext=(8, 8), textcoords='offset points',color='C3',size=15)#, weight='bold')

        plt.show()


        happy = messagebox.askyesno("","Are you happpy with the result?")
    plt.close(1)

    
    return coords  
    

# =============================================================================
# Seccion de los datos del tamano de la apertura    
# =============================================================================
def SecImData(ap, data, method, subpixels=None):
      mask=ap.to_mask(method=method,subpixels=subpixels)
      mask_data=mask.data
      sec=mask.cutout(data)
      sec_weight=sec*mask_data
      sec_data = sec_weight[mask_data>0]
      # Quitamos los valores NaN
      sec_data = sec_data[np.isfinite(sec_data)]
      
      return sec_data



class AnnSky:
    
    def __init__(self,data, coord, r_in, r_out, method, subpixels=None):
        from photutils import CircularAnnulus

        self.ap  = CircularAnnulus(coord, r_in=r_in, r_out=r_out)
    
        self.sec_data = SecImData(self.ap, data, method=method, subpixels=subpixels)
        # media sigma clip
        
    def stat(self, sigma_clip):
        from astropy.stats import sigma_clipped_stats        
        mean, median, stddev = sigma_clipped_stats(self.sec_data, sigma=sigma_clip)
        
        return mean, median, stddev
    
    def plot(self, color= 'C0', ls='solid',lw=1):
        self.ap.plot(color=color, ls =ls,lw=lw)



def FWHM(data,xc,yc):
    from astropy.modeling.models import Gaussian2D
    from astropy.modeling.models import Const2D

    from astropy.modeling.fitting import LevMarLSQFitter

    # Data:
    x, y = np.mgrid[:data.shape[1], :data.shape[0]]
    
    # Model:
    cte0 = np.nanmean(data)
    Cte = Const2D(cte0)
    
    Gauss = Gaussian2D(amplitude=data[int(yc),int(xc)], x_mean=xc, y_mean=yc)
    # Parametros fijos
    Gauss.x_mean.fixed = True
    Gauss.y_mean.fixed = True

    model = Gauss+Cte

    # Parametros ligados: imponemos que sea una gaussiana redonda
    def tie(model):
        return model.y_stddev_0
    
    model.x_stddev_0.tied = tie

    # Fit
    fitter = LevMarLSQFitter()
    import warnings
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitter(model, x, y, data)
    
    sigma = fit.x_stddev_0.value
 
    FWHM = 2.3548 * sigma
    
    return FWHM



def Seeing(data, r = 11):
    
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder     
    from astropy.nddata.utils import Cutout2D

    # Source detection

    mean, median, std = sigma_clipped_stats(data, sigma=3) 

    starfind = DAOStarFinder(fwhm=4.0, threshold=5.*std, exclude_border=True, sky=median) 
    sources = starfind(data-median) 

    x = sources['xcentroid']    
    y = sources['ycentroid']
    
    # FWHM over all the detected sources
    fwhm = []
    for i in range(len(x)):    
        cut = Cutout2D(data, [x[i],y[i]], r, mode='partial')
        sec = cut.data
        xc , yc = cut.to_cutout_position([x[i],y[i]])
        fwhm.append(FWHM(sec,xc,yc))
    
    return np.array(fwhm)


"""

    # Magnitud fumental
    # Asignamos el mismo zeropoint que iraf
    
    inst_mag = -2.5*np.log10((phot_table['aperture_sum'] - bkg_sum)/exptime[i])
    
    # Errores segun los calcula IRAF

    flux = phot_table['aperture_sum'] - bkg_sum # cuentas debidas solo a la senal
    epadu = gain[i] # ganancia
    
    area = aperture.area
    stdev = bkg_std # desviacion estandard del cielo
    nsky = annulus_apertures.area
    error = np.sqrt(flux/epadu + area*stdev**2 + area**2*stdev**2/nsky)
    merr.append(1.0857*error/flux)    


"""













    