#ifndef FULL_H
#define FULL_H

void advanced_dijkstra(int width, int frame_accurarcy, float *cost, int m, int n, int** tour, int* tour_len, int** box, int* box_len);
void plain_dijkstra(int width, int frame_accurarcy, float *cost, int m, int n, int** tour, int* tour_len);
void accumulate(float *cost, int m, int n);
#endif