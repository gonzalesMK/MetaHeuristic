.. _parallel:

=============
Parallel Processing
=============

The parallel feature is given by the multiprocessor package.

When the option parallel=True is set, the line below is called

>> from multiprocessing import Pool

This changes only the evaluation part of individuals on algorithms based in population,
changing map to Pool().map, with default inputs. The others algorithms have just
the initialization of the inicial population, if more than one individual, implemented with
map.

Check the example. 