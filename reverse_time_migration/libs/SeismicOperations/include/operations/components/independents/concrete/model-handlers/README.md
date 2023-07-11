# Model handlers

All different implementations of the model handlers interface should reside here. Description of the different
implementations should be below.

for the homogenous model handler , the user puts the required values in file named input.txt whose structure is as
follow:

nx,nz,ny,dx,dz,dy| velocity,Upper left point.x ,Upper left point.z,Upper left point.y, lower right point.x ,lower right
point.z, lower right point.y| window upper left point .x,window upper left point .x ,window upper left point .x ,lower
right point .x ,lower right point .z , lower right point .y|

as an example :

3,3,3,0.01,0.01,0.01| 200,0,0,0,3,3,3| 0,0,0,3,3,3|

in the case of 2D , you replace each value for ny by 1, dy by 0 and lower right.y by 1 , so the example will be :

3,3,1,0.01,0.01,0| 200,0,0,0,3,3,1| 0,0,0,3,3,1|

also another row can be added after , to include density values in case of frist order.

muliple values of velocities can be included in this file . to add another velocity layer , you first right the velocity
and then the starting point (x,z,y) (upper left point ) and the end point (x,z,y)(lower right point)

example for several layers :

above line 200,0,0,0,1,1,1,300,2,2,2,3,3,3| rest of lines

