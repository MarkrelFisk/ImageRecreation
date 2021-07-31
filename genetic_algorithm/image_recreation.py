import numpy as np
import matplotlib.pyplot as plt
import cv2

class genetic_sim: # class object of recreated image
    
    def __init__(self, input_im, maxit, tol, num_objects):           
        self.target = input_im                                                          # Target image
        self.maxit = maxit                                                              # Max number of generations
        self.tol = tol                                                                  # fitness tolerance

        self.it = 0                                                                     # Initial iteration
        self.alpha = 0.8                                                                # Opacity
        self.image = np.ones((input_im.shape[0], input_im.shape[1], 3), np.uint8) * 255 # Recreated image
        
        # Create random objects
        self.x = np.random.randint(0,high = self.image.shape[0], size = num_objects)
        self.y = np.random.randint(0, high = self.image.shape[1], size = num_objects)
        self.radius = np.random.randint(10, high = np.min([self.image.shape[0],self.image.shape[1]])//2, size = num_objects)
        self.color = np.random.randint(0, high=256, size=(num_objects,3)).tolist()
        #self.color = (255 - (255 - np.random.randint(0, high=256, size=(num_objects,3))) * self.alpha).tolist()

    #def fitness(self): 
        #copy = self.image.copy()
        #mask = np.zeros(self.image.shape[:2],dtype = 'uint8')
        #cv2.circle(mask, (self.x[0], self.y[0]), self.radius[0], 255, -1)
        #masked_image = cv2.bitwise_and(copy, copy, mask=mask)
        #masked_target = cv2.bitwise_and(self.target,self.target,mask=mask)
        
        #plt.imshow(masked_image)
        #plt.show()

        #plt.imshow(masked_target)
        #plt.show()

        #np.sum( np.abs(self.image - self.target, axis = 1)
        #return np.abs(np.sum(self.color.ravel()-self.target[].ravel())) # fitness definition

    def create_pheno(self):
        
        image = np.ones((self.target.shape[0], self.target.shape[1], 3), np.uint8) * 255

        for i in range(len(self.x)):
            copy = image.copy()
            cv2.circle(image,(self.x[i],self.y[i]),self.radius[i],self.color[i],-1)
            image = cv2.addWeighted(copy, self.alpha, image, 1-self.alpha, gamma=0)
        
        self.image = image

        return

    def draw_recreation(self):

        self.create_pheno()

        plt.imshow(self.image)
        plt.show()

        return 

    def update_gen(self):

        x = np.random.randint(0,high = self.image.shape[0])
        y = np.random.randint(0, high = self.image.shape[1])
        radius = np.random.randint(5,high = np.min([self.image.shape[0],self.image.shape[1]])//2)
        color = np.random.randint(0, high=256, size=(3, )).tolist()
        copy = self.image.copy()
        proposed_image = cv2.addWeighted(cv2.circle(copy,(x,y),radius,color,-1), self.alpha, copy,1-self.alpha, gamma=0)
        
        if self.fit >= self.fitness():
            self.fit = self.fitness()
            self.image = proposed_image
            self.draw_recreation()
        return

    def print_status(self):
        print(f'Generation: {self.it}')
        print(f'fitness: {self.fit}')
        return
    
    def simulate(self):
    
        self.fit = self.fitness()*2

        while self.it < self.maxit and self.fit >= self.tol:
            self.update_gen()
            self.it += 1
            self.print_status()

        return

if __name__ == '__main__':
    im = cv2.circle(np.ones((100, 100, 3), np.uint8) * 255,(50,50),20,(0,0,255),-1) 
    #plt.imshow(im)
    #plt.show()
    test_simulation = genetic_sim(im,10,0.1,10)
    test_simulation.draw_recreation()
    #test_simulation.fitness()
    fig, ax = plt.subplots(3)
    ax[0].imshow(test_simulation.target)
    ax[1].imshow(test_simulation.image)
    ax[2].imshow(np.abs(np.sum(test_simulation.target,axis=1)-np.sum(test_simulation.image,axis=1)))
    plt.show()

#    print(test_simulation.maxit)
#    test_simulation.simulate()
#    test_simulation.draw_recreation()

