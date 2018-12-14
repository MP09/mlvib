import numpy as np

from mlvib.gaussian_process import Gaussian_process
from mlvib.kernels import RBF

import matplotlib.pyplot as plt

class Active_Learner:

    def __init__(self, func, num_inputs, input_dim, region):
        """
        Parameters:
        -- func: 
           Function to learn
        -- num_inputs: int 
           Number of inputs to the function to learn.
        -- input_dim: int
           Dimension of each input to func. 
        -- region: List of tuples
           (min, max) for each input dimension, specifying the region of which to learn the function

        (num_inputs = 2, input_dim = 1 corresponds to a 2D-surface)
        """

        # Handle inputs:
        self.func = func
        self.num_inputs = num_inputs
        self.input_dim = input_dim
        self.region = np.array(region)

        # Setup GPR:
        self.kernel = RBF(length_scale=1)
        self.initialize()

    def initialize(self, num_points=2):
        """
        Initialize the GP.
        """
        X = np.zeros((num_points, num_inputs))
        y = np.zeros((num_points, 1))
        for ii in range(num_points):
            # Need two points so add also a random point:
            X[ii, :] = (np.random.rand(self.num_inputs)*(self.region[:, 1]-self.region[:, 0])+
                self.region[:, 0])
            y[ii, :] = self.func(*X[ii, :])
        
        self.GP = Gaussian_process(X, y, kernel=self.kernel)
        self.GP.fit()

    def add_random(self):
        """
        Adds a randomly chosen point to the GP.
        """
        X = (np.random.rand(self.num_inputs)*(self.region[:, 1]-self.region[:, 0])+
             self.region[:, 0])
        y = self.func(*X)
        self.GP.add_point(X, y)
        self.GP.fit()
        
    def choose(self, num_points, grid_size=500):
        """
        Choose which points to add to the GP based on maximum variance.

        Parameters:
        -- num_inputs: int
        """


        # Make grid on which to calculate variance:
        G = [np.linspace(*region, grid_size) for region in self.region]
        grid = np.array(np.meshgrid(*G))
        grid = grid.reshape(self.num_inputs, grid_size**self.num_inputs).T

        # Calculate variance on grid:
        var = self.GP.variance(grid)
        
        # Pick the maximum:
        ind = np.argpartition(var, -num_points)[-num_points:]

        # Points:
        x = grid[ind, :]
        y = np.array([[self.func(*xx) for xx in x]]).T

        # Add points to GP:
        self.GP.add_point(x, y)

        # Fit to new points:
        self.GP.fit()
        
    def learn(self, criteria=0.1, grid_size=50, smart=True):
        """
        Learn the function in a variance-greedy way.
        """

        # Grid:
        G = [np.linspace(*region, grid_size) for region in self.region]
        X = np.array(np.meshgrid(*G))
        X = X.reshape(self.num_inputs, grid_size**self.num_inputs).T

        # Evaluate function:
        y = np.array([self.func(*x) for x in X]).T
        
        count = self.GP.n
        state = True
        while state:

            if smart:
                self.choose(1)
            else:
                self.add_random()
            error = self.GP.score(X, y)
            score = np.mean(error)
            idx = np.argmax(error)
            
            state = score > criteria
            count += 1
            #if count % 50 == 0:
                #self.visualize()

#        self.visualize()
                
        return count

    def visualize(self):

        if num_inputs == 1:
            self.visualize1D()
        elif num_inputs == 2:
            self.visualize2D()
        else:
            print('I dont know how to do that')
            
    def visualize1D(self):
        
        x = np.linspace(*self.region[0], 300).reshape(300, 1)
        y = self.func(x)
        
        # GP prediction:
        yp, std = self.GP.predict(x, return_std=True)


        plt.scatter(self.GP.X.flatten(), self.GP.y.flatten(), color='black')
        plt.plot(x, y, color='black')
        plt.plot(x, yp, '--', color='red')
        fac = 1.95
        plt.fill_between(x.flatten(), y.flatten()-1.95*std.flatten(), y.flatten()+1.95*std.flatten(), color='purple', alpha=.5)

        plt.tight_layout
        plt.show()

        
    
    def visualize2D(self):
        """
        Visualize the learning process

        Only works in 2D
        """

        from mpl_toolkits.mplot3d import Axes3D

        grid_size = 50
        G = [np.linspace(*region, grid_size) for region in self.region]
        X = np.array(np.meshgrid(*G))
        X = X.reshape(self.num_inputs, grid_size**self.num_inputs).T

        # Evaluate function:
        y = np.array([self.func(*x) for x in X]).T

        # GP prediction:
        yp = self.GP.predict(X, return_std=False)

        
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.scatter(X[:, 0], X[:, 1], y, alpha=0.5, color='red')
        ax.plot_trisurf(X[:, 0], X[:, 1], yp, alpha=0.5)

        plt.show()
        

        
            

if __name__ == '__main__':

#    np.random.seed(100)

    # 2D Example:
    func = lambda x, y: x*np.sin(np.cos(x)) + y*np.cos(x)
    num_inputs = 2
    input_dim = 1
    region = [(-10, 10), (-10, 10)]
    
    # 1D Example:
    #func = lambda x: np.sin(np.cos(x))*x
    #num_inputs = 1
    #input_dim = 1
    #region = [(-20, 20)]


    samples = 10
    smart_count = np.zeros(samples)
    dumb_count = np.zeros(samples)
    for jj in range(samples):
        
        AL = Active_Learner(func, num_inputs, input_dim, region)
        smart_count[jj] = AL.learn(smart=True)

        AL = Active_Learner(func, num_inputs, input_dim, region)
        dumb_count[jj] = AL.learn(smart=False)

    print(np.mean(smart_count))
    print(np.mean(dumb_count))
