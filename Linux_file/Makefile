FC = gfortran
FFLAGS = -O2 -std=legacy -fallow-argument-mismatch

hijing.o: hijing.f
	$(FC) $(FFLAGS) -c hijing.f -o hijing.o

hipyset.o: hipyset.f
	$(FC) $(FFLAGS) -c hipyset.f -o hipyset.o

test.o: test.f
	$(FC) $(FFLAGS) -c test.f -o test.o

test: test.o hijing.o hipyset.o
	$(FC) -o test test.o hijing.o hipyset.o
