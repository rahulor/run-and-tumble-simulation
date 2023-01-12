import numpy as np
class DimensionTwo:
    def __init__(self):
        from inputs import food_surface, horizon, conc_surface, conc_horizon, grad_fun
        self.conc_min = conc_horizon
        self.conc_max = conc_surface
        self.xmin     = food_surface
        self.xmax     = horizon
        
        self.gamma = (self.conc_max - self.conc_min)/(self.xmax-self.xmin) # linear conc gradient
        self.beta = (1/(self.xmax-self.xmin))*np.log(self.conc_max/self.conc_min) # exponential decay const 
        self.origin = np.array([0, 0])
        self.grad_fun = grad_fun
        self.set_text_info()
        self.set_conc_fun()
        
    def set_text_info(self):
        self.text_line      = f'{"":-<40} \n'
        self.text_cx0       = f'c(x0) = {self.conc_max:<10} [\mu M] \n'
        self.text_x0        = f'x0    = {self.xmin:<10} [\mu m] \n'
        self.text_gamma     = f'gamma = {self.gamma:<10,.2f} [\mu M/\mu m] \n'
        self.text_beta      = f'beta  = {self.beta:<10,.4f} [1/\mu m] \n'
        
    def set_conc_fun(self):
        if self.grad_fun == 'lin':
            self.conc = lambda x: self.conc_max - self.gamma*(x-self.xmin)
            text_eqn_name   = 'linear\n'
            text_eqn        = 'c(x)  = c(x0) - gamma*(x-x0) \n'
            self.conc_txt   = (f'{text_eqn_name}{self.text_line}{text_eqn}{self.text_line}' + 
                               f'{self.text_x0}{self.text_cx0}{self.text_gamma}' )
        elif self.grad_fun == 'exp':
            self.conc = lambda x: self.conc_max * np.exp(-self.beta*(x-self.xmin))
            text_eqn_name   = 'exponential\n'
            text_eqn        = 'c(x)  = c(x0)*exp(-beta(x-x0)) \n'
            self.conc_txt   = (f'{text_eqn_name}{self.text_line}{text_eqn}{self.text_line}' + 
                               f'{self.text_x0}{self.text_cx0}{self.text_beta}' )
        elif self.grad_fun == '1/r':
            self.conc = lambda x: self.conc_max*self.xmin / x
            text_eqn_name   = 'inverse r\n'
            text_eqn        = 'c(x)  = c(x0)*x0/x \n'
            self.conc_txt   =  (f'{text_eqn_name}{self.text_line}{text_eqn}{self.text_line}' + 
                               f'{self.text_x0}{self.text_cx0}' )
        else:
            raise ValueError('check grad_fun: accept only {lin, exp, 1/r}')

    def concentration_at(self, position):
        dist_vector = position - self.origin
        radial_dist = np.linalg.norm(dist_vector) # distance to the [position] from [food-center].
        flag_inside = False
        flag_outside = False
        if radial_dist <= self.xmin:
            c = self.conc_max
            flag_inside = True
        elif radial_dist > self.xmax:
            c = 0
            flag_outside = True
        else:
            c = self.conc(radial_dist)
        return(max(c, 0), flag_inside, flag_outside)
    
         
if __name__ == "__main__":
    pass