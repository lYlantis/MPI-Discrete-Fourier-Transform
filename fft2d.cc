// Distributed two-dimensional Discrete FFT transform
// Jacob Ashmore
// ECE4122 Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

int numtasks, rank, rc, chunks;

void Transform1D(Complex* h, int w, Complex* H)
{

  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.

	Complex* new_h = h;
	Complex* new_H = H;
	for(int N=0; N<w; N++) {

		for(int k=0; k<w; k++) {

			Complex W(cos(2*M_PI*N*k/w),-sin(2*M_PI*N*k/w));
			*H=(*H)+(*h)*W;
			h++;

		}
		h = new_h;
		H++;

	}
	H = new_H;

}

//function to transpose
void transpose(Complex* before_t, Complex* after_t, int w, int h) {
	for(int j=0; j<h; j++) {
		for(int i=0;i<w;i++) {
			after_t[i*w+j]=before_t[j*h+i];
		}
	}
}

//cpu1-cpu15 send all of their data immediately to cpu0 as soon as it becomes available
void send_to_cpu0(int width, int height, Complex* c2_data_t) {
	if(rank!=0) {
		MPI_Request request;
		rc = MPI_Isend(c2_data_t, (2*height*width)/chunks, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &request);						//non-blocking send from cpu1-cpu15, should send only their chunk size (8192) to cpu0
		if (rc != MPI_SUCCESS)
			{
				cout << "Rank " << rank << " send failed, rc " << rc << endl;
			}
		}

	//cpu0 receives all of the data from the other processes in blocking receive mode
	if(rank==0) {
		for(int j=1; j<chunks; j++) {		//only receiving from cpu1-cpu15
			MPI_Status status;
			rc = MPI_Recv((c2_data_t+(height*width*j/chunks)), 2*(height*width)/chunks, MPI_COMPLEX, j,0, MPI_COMM_WORLD, &status);				//blocking receive for cpu0, place in order based on what cpu it came from
			if (rc != MPI_SUCCESS)
			  {
				cout << "Rank " << rank << " recv failed, rc " << rc << endl;
			  }
		}
	}
}

void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.

	InputImage image(inputFN);  // Create the helper object for reading the image
	int width = image.GetWidth();
	int height = image.GetHeight();

  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is

	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data

	Complex* c_data = image.GetImageData();				//point to the beginning of the array h[k]
	Complex* c2_data = new Complex [width*height];		//new empty array of size 256*256= 65,536


  // 5) Do the individual 1D transforms on the rows assigned to your CPU

	chunks = height/numtasks;							//256/16=16 for this case
	int start_row = height*rank/chunks;					//cpu0 starts at row 0, cpu1 starts at row 16, cpu2 starts at row 32 .... cpu15 starts at row 240

	for(int j=0; j<chunks; j++) {										//each of the 16 cpus does 16 rows

		int offset = width*(start_row+j);								//Where along the array does each cpu start? cpu0 starts at element 0, cpu1 starts at element 256*(16+j)=4096,... cpu15 starts at element 256*(240+j)=61,440
		Transform1D( (c_data+offset), width, (c2_data + (width*j)) );			//c_data(h[0]) + offset= 0,4096,...61,440, c2_data(H[0]) + (256*j)) = 0,1...16

	}

  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
	Complex* c2_data_trans = new Complex[width*height];
	send_to_cpu0(width, height, c2_data);
	if(rank==0){
		image.SaveImageData("MyAfter1D.txt", c2_data, width, height);
		transpose(c2_data, c2_data_trans, width, height);
	}

  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.


  // 7) Receive messages from other processes to collect your columns

	if(rank == 0) {
		for(int l=1;l<chunks;l++){
			MPI_Request request;
			rc = MPI_Isend(c2_data_trans, (2*height*width), MPI_COMPLEX, l, 0, MPI_COMM_WORLD, &request);
			if (rc != MPI_SUCCESS)
		{
			cout << "Rank " << rank << " send failed, rc " << rc << endl;
		}
	}
	} if(rank!=0) {
		MPI_Status status;
		rc = MPI_Recv(c2_data_trans, (2*height*width), MPI_COMPLEX, 0,0, MPI_COMM_WORLD, &status);
		if (rc != MPI_SUCCESS)
		{
			cout << "Rank " << rank << " recv failed, rc " << rc << endl;
		}
	}

  // 8) When all columns received, do the 1D transforms on the columns
	Complex* c2_data_2D = new Complex[height*width];
	start_row = height*rank/chunks;
	for(int j=0; j<chunks; j++) {										//each of the 16 cpus does 16 rows

		int offset = width*(start_row+j);
		Transform1D((c2_data_trans+offset), width, (c2_data_2D+(width*j)));
	}

  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().

	send_to_cpu0(width, height, c2_data_2D);


	if(rank==0) {
		Complex* c2_data_Final = new Complex[width*height];
		transpose(c2_data_2D, c2_data_Final, width, height);
		image.SaveImageData("MyAfter2D.txt", c2_data_Final, width, height);
	}

}



int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line

  // MPI initialization here
  rc = MPI_Init(&argc,&argv);
  if (rc != MPI_SUCCESS) {
      printf ("Error starting MPI program. Terminating.\n");
      MPI_Abort(MPI_COMM_WORLD, rc);
    }

  Transform2D(fn.c_str()); // Perform the transform.

  // Finalize MPI here
  MPI_Finalize();
}  
  

