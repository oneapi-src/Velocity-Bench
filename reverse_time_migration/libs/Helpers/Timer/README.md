# Timer
<p>
  <img src="https://img.shields.io/pypi/status/Django.svg" alt="stable"/>
</p>

## Usage

####  Building Timer

if you want to use the timer in your ```.cpp``` file, let's call it sample.cpp, you would issue the following command to build the application:

``` shell script
mpiicpc -o sample -std=c++11 sample.cpp timer.cpp -qopenmp 
```

note that any application using the timer needs to be built
with ```mpiicpc``` because it supports timing multiple processes using the MPI framework. Make sure that you also include "Timer.h" in your application.

#### Using Timer
Timer follows the singleton design pattern, the object can be initialized anywhere in the code and it will remember the data initialized in an older object. this is how the timer is initialized for single process use:

```c++
Timer* t = Timer::GetInstance();
```
                      
for multiple processes it would be initialized like this:
```c++
Timer* t = Timer::getInstance(1);
```
                      
Note that the multiple processes feature is not yet
fully tested Make sure that you point the pointer to the return value of the
static method GetInstance(), merely initializing it like this: Timer* t is not
enough.

To start timing a function the following method is called
```c++
t->start_timer(function_name) 
```
before the function call, then
```c++
t->stop_timer(function_name) 
```
after the function call. For example the following code would be used to time a function called func1.
```c++
t->start_timer("func1");
func1();
t->stop_timer("func1");
```

<br>
               
The string passed to the start_timer and stop_timer functions
doesn't have to be the real function name, any string would suffice but it is
generally recommended to used the actual function read to understand the report
better. Just make sure that the strings passed to the start_timer function is
the same string passed to the stop_timer function when you intend to stop that
timer.

<br>

Retrieving the timing report can be done in 3 ways:
* Printing to ```std::cout```
```c++
t->print_report();
```

* Exporting to a file
```c++
t->export_to_file(filename);
```

* Returning the report as a string
```c++
t->get_report();
```

<br>

If you want to print the report data in scientific notation
instead of its current format then line 184 should be modified, remove
"std::fixed" from the stream. The precision of the numbers

