#include "sys/time.h"

extern double elapsedTime (void)
{
	struct timeval t;
	gettimeofday(&t, 0);
	return ((double)t.tv_sec +((double)t.tv_usec / 1000000.0));
}