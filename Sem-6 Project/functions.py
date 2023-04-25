from file_fn import *

#Interpolation begin
class BilinearInterpolation:
    def __init__(self,dataset):
        self.data = dataset
        self.size = dataset.shape[0] - 1
        self.size_of_division = 1/self.size

    def point(self,x):
        x_2 = (np.ceil(x/self.size_of_division))
        if x_2==0:
            x_2 += 1
        return np.array([x_2-1,x_2])

    def finding_point(self,x_i,y_i):
        point_x = self.point(x_i)
        point_y = self.point(y_i)
        point_x = np.array(point_x,dtype = int)
        point_y = np.array(point_y,dtype = int)
        values = np.zeros((2,2))

        row = 0
        for i in point_x:
            column = 0
            for j in point_y:
                values[row][column] += self.data[i][j]
                column+=1
            row+=1

        return point_x*self.size_of_division,point_y*self.size_of_division,values

    def predict(self,x_i,y_i):    
        x, y, F = self.finding_point(x_i, y_i)
        H = 1/((x[1]-x[0])*(y[1]-y[0]))
        X = np.array([x[1]-x_i,x_i-x[0]])
        Y = np.array([y[1]-y_i,y_i-y[0]])
        res = H*(np.matmul(np.matmul(X.T,F),Y))
        return res

class TrilinearInterpolation:

    def __init__(self,dataset):
        self.data = dataset
        self.size = dataset.shape[0] - 1
        self.size_of_division = 1/self.size

    def point(self,x):
        x_2 = (np.ceil(x/self.size_of_division))
        if x_2==0:
            x_2 += 1
        return np.array([x_2-1,x_2],dtype = int)

    def predict(self,xi,yi,zi):
        if zi > 1.:
            return self.predict(xi,yi,1)
        if zi < 0:
            return self.predict(xi,yi,0)
        
        if xi > 1 or xi < 0 or yi > 1 or zi > 1 or zi < 0 or yi < 0:
            return -1

        z = self.point(zi)
        f1_interpolate = BilinearInterpolation(dataset=self.data[:][:][z[0]])
        f2_interpolate = BilinearInterpolation(dataset=self.data[:][:][z[1]])
        f1 = f1_interpolate.predict(xi, yi)
        f2 = f2_interpolate.predict(xi, yi)
        z = self.size_of_division * z
        h = (zi - z[0]) * (f2 - f1) / (z[1] - z[0])
        p = f2 + h

        datastemp = self.data
        nx = np.shape(datastemp)[0]
        ny = np.shape(datastemp)[1]
        nz = np.shape(datastemp)[2]
        
        deltx = 1.0/nx
        delty = 1.0/ny
        deltz = 1.0/nz
        n0_x = np.int(np.floor((xi-0.5*deltx)/deltx))
        x0 = (n0_x+0.5)*deltx 
        x1 = x0+deltx
        n1_x = n0_x+1
        n0_y = np.int(np.floor((yi-0.5*delty)/delty))
        y0 = (n0_y+0.5)*delty 
        y1 = y0+delty
        n1_y = n0_y+1 
        n0_z = np.int(np.floor((zi-0.5*deltz)/deltz))
        z0 = (n0_z+0.5)*deltz 
        z1 = z0+deltz
        n1_z = n0_z+1
                
        xd = (xi-x0)/(x1-x0)
        yd = (yi-y0)/(y1-y0)
        zd = (zi-z0)/(z1-z0)

        if(n0_x>255 or n0_y>255 or n0_z>255 or n1_x>255 or n1_y>255 or n1_z>255):              
            return -1

        c000 = datastemp[n0_x,n0_y,n0_z]
        c100 = datastemp[n1_x,n0_y,n0_z]
        c010 = datastemp[n0_x,n1_y,n0_z]
        c001 = datastemp[n0_x,n0_y,n1_z]
        c110 = datastemp[n1_x,n1_y,n0_z]
        c101 = datastemp[n1_x,n0_y,n1_z]
        c011 = datastemp[n0_x,n1_y,n1_z]
        c111 = datastemp[n1_x,n1_y,n1_z]

        c00 = c000*(1.0-xd)+c100*xd
        c01 = c001*(1.0-xd)+c101*xd
        c10 = c010*(1.0-xd)+c110*xd
        c11 = c011*(1.0-xd)+c111*xd

        c0 = c00*(1.0-yd)+c10*yd
        c1 = c01*(1.0-yd)+c11*yd

        p = c0*(1.0-zd)+c1*zd

        return p
        
#Interpolation End

#Runge-Kutta begin
class MultivariableRungaKutta:
    def __init__(self,dataset_x,dataset_y,dataset_z,iterations = 1000):
        self.iterations = iterations # number of iterations we want to perform
        self.data_x = dataset_x
        self.data_y = dataset_y
        self.data_z = dataset_z
        self.interpolate_x = TrilinearInterpolation(dataset_x)
        self.interpolate_y = TrilinearInterpolation(dataset_y)
        self.interpolate_z = TrilinearInterpolation(dataset_z)

    ''' This function takes one step in runge kutta and gives us 
        the next value of x,y,z,l, generated_error'''
    def predict(self,li,x_vector):

        l = li
        x = x_vector[0]
        y = x_vector[1]
        z = x_vector[2]

        value = []
        value.append(self.interpolate_x.predict(x,y,z))
        value.append(self.interpolate_y.predict(x,y,z))
        value.append(self.interpolate_z.predict(x,y,z))
        return np.array(value)

    def one_step(self,x0,y0,z0,l0,h_try,interpolation):
        '''Values specified in the research paper for adaptive size runge kutta '''
        a = [0, 1 / 5, 3 / 10, 3 / 5, 1, 7 / 8]
        b = [[0, 0, 0, 0, 0],
             [1 / 5, 0, 0, 0, 0],
             [3 / 40, 9 / 40, 0, 0, 0],
             [3 / 10, -9 / 10, 6 / 5, 0, 0],
             [-11 / 54, 5 / 2, -70 / 27, 35 / 27, 0],
             [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096]]
        c = [37 / 378, 0, 250 / 621, 125 / 594, 0, 512 / 1771]
        c_star = [2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4]

        ''' Intializaling the x,y,z,l'''
        xi = x0
        yi = y0
        zi = z0
        li = l0

        ''' finding k1,k2... k6 as per the research paper'''
        k1 = h_try * self.predict(li, [xi, yi, zi])
        k2 = h_try * self.predict(li + a[1] * h_try, [xi, yi, zi] + b[1][0] * k1)
        k3 = h_try * self.predict(li + a[2] * h_try, [xi, yi, zi] + b[2][0] * k1 + b[2][1] * k2)
        k4 = h_try * self.predict(li + a[3] * h_try, [xi, yi, zi] + b[3][0] * k1 + b[3][1] * k2 + b[3][2] * k3)
        k5 = h_try * self.predict(li + a[4] * h_try, [xi, yi, zi] + b[4][0] * k1 + b[4][1] * k2 + b[4][2] * k3 + b[4][3] * k4)
        k6 = h_try * self.predict(li + a[5] * h_try, [xi, yi, zi] + b[5][0] * k1 + b[5][1] * k2 + b[5][2] * k3 + b[5][3] * k4 + b[5][4] * k5)

        ''' resulted x,y,z,l value'''
        vxi = np.array([xi, yi, zi])
        vxi = vxi + c[0] * k1 + c[1] * k2 + c[2] * k3 + c[3] * k4 + c[4] * k5 + c[5] * k6
        li = li + h_try

        ''' Calculating the error generated after taking the step '''
        x_error = (c[0] - c_star[0]) * k1 + (c[1] - c_star[1]) * k2 + (c[2] - c_star[2]) * k3 + (c[3] - c_star[3]) * k4 + (c[4] - c_star[4]) * k5 + (c[5] - c_star[5]) *k6

        return li, vxi, x_error

    ''' This function tells us the step we took is valid or not.In this 
        case valid means that fraction  is less than one '''
    def error_estimation_step(self, x0, y0, z0, l0, h0,interpolation):
        # intializations of the parameters
        SAFETY = 0.9
        P_shrink = -0.25
        P_groww = -0.2

        # We can change the desired error as per the requirement
        error_correction = 1.89e-4 # value is according to research paper

        ''' Intializing x,y,z,l'''
        h = h0
        li = l0
        xi = x0
        yi = y0
        zi = z0
        trials = 0
        # checking if step is valid or not
        while trials < 50:

            l_next, x_next, x_error = self.one_step(xi,yi,zi,li,h,interpolation)
 
            desired_error = 1.0E-6*h*abs(self.predict(l_next,x_next))
            for i in range(len(desired_error)):
                if (desired_error[i]<1.0E-19):
                    desired_error[i]=1.0E-19

            frac_error = -1 # intialising the value of fraction_error at its minimum

            for i in range(len(x_error)):
                frac_error = max(abs(x_error[i]/ desired_error[i]),frac_error) # finding max fraction_error in the step

            h_prev = h
            ''' 
            fraction error comes out to be greater than one then we have
            to decrease the step size as larger step size might decrease 
            the accuracy of the curve.
            '''

            if frac_error >= 1.:

                h = SAFETY * h_prev * (frac_error ** P_shrink)
                ''' 
                But there is limit till we can decrease the 
                value of h  specified as per the research paper.
                '''
                if h < 0.1 * h_prev:
                    h = 0.1 * h_prev
            else:
                ''' As fraction error is less than then we can increase the step size 
                by reduce the computational cost. Still there is limit to that as well. '''
                if frac_error > error_correction:
                    h_next = SAFETY * h * (frac_error ** P_groww)
                else:
                    h_next = 5. * h
                return l_next, x_next, h_next # returning the next set of values

            trials+=1
        print("Number of trails of calculating h exceeded, try with different intial points")
        return np.inf,[],0. 

    #This functions runs above function n_iterations to give the desired results
    def variable_step_size_approximation(self,l0,x0,y0,z0,h0,interpolation = True):
        # for storing the output values
        l = list()
        x = list()
        y = list()
        z = list()
        H = []
        iteration = []

        # intialization for runge kutta method
        
        li = l0
        xi = x0
        yi = y0
        zi = z0
        h = h0
        for i in range(self.iterations):
            l.append(li)
            x.append(xi)
            y.append(yi)
            z.append(zi)
            H.append(h)
            iteration.append(i)

            li, w, h = self.error_estimation_step(xi, yi, zi, li, h , interpolation)
            if li == np.inf:
                break
            xi,yi,zi = w

            if xi > 1 or xi < 0 or yi > 1 or zi > 1 or zi < 0 or yi < 0:
                break

            print("x,y,z = ",xi,yi,zi)

        return l,x,y,z,H,iteration # returning the result






                


                






