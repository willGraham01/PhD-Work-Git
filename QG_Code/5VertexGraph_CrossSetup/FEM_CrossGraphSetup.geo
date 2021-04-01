/* https://gmsh.info/doc/texinfo/gmsh.html */

/*PLAN:
1) Define the boundary nodes
2) Draw the boundary lines, making note of the ones that need to be periodically linked
3) Define the boundary loop, BUT NOT the surface
4) Define a single "hole"
5) Translate "hole", making list of IDs
6) Create surface using boundary loop minus copies of holes. */

/*Things to setup before we begin meshing...*/
N = 4;
/* N details the number of unit cells in x and y, so N^2 period cells constitute the entire domain */
If ( N<2 )
    Error("N==%.0f - MESH CREATION WILL FAIL, ABORT", N);
    Abort;
EndIf
h = 0.5 / N; //tube widths
alpha = 2.0; //parameter alpha that appears as the coupling constant in the QG problem. NOTE: This assumes alpha = V_edge / V_junction, so might be 1/alpha when reconciled with analytic work!
r = Sqrt( h / (alpha * Pi) ); //circle radius, given alpha and tube width
cLen = h / 2.0; //Characteristic length for mesh points - simplex size will be of this order
circToTube = Sqrt( (r*r) - ((h*h)/4.0) ); //horizontal/vertical distance from centre of a circle to the beginning of a tube
Printf("Domain has %.0f unit cells.", N);
Printf("Circle radius %.5f. Tube widths are %.5f", r, h);
Printf("circToTube has value %.5f", circToTube);

/*The list boundaryStarts will record the indices at which one piece of the exterior boundary begins. boundaryStarts(i) is the index of the point which begins the i-th piece of the exterior boundary */
boundaryStarts = {};
/*The variable j will be our counting variable for points*/
j = 0;

/*
Point( index )  = { x, y, z, mesh_size}
*/

/*Begin with the nodes on the bottom boundary, IE along y==0*/
For i In {0 : N-1}
    Point(j) = { i, 0.5 - h/2.0, 0, cLen};                              //Leftmost tube end
    Point(j+1) = { i + 0.5 - circToTube, 0.5 - h/2.0, 0, cLen};         //Left tube joins circle
    Point(j+2) = { i + 0.5 - h/2.0, 0.5 - circToTube, 0, cLen};         //Bottom tube joins circle on left
    Point(j+3) = { i + 0.5 - h/2.0, 0, 0, cLen};                        //Bottom-left tube corner
    Point(j+4) = { i + 0.5 + h/2.0, 0, 0, cLen};                        //Bottom-right tube corner
    Point(j+5) = { i + 0.5 + h/2.0, 0.5 - circToTube, 0, cLen};         //Bottom tube joins circle on right
    Point(j+6) = { i + 0.5 + circToTube, 0.5 - h/2.0, 0, cLen};         //Right tube joins circle
    j = j+7;    //update tracker for next cell
EndFor
/*Have to manually add the final node in preparation for making the right boundary */
Point(j) = {N, 0.5 - h/2.0, 0, cLen};
j = j+1; //update tracker with info
boundaryStarts += {j+1}; //the right boundary starts at the next point we create, which should be the index j+1

/*Begin nodes on the right boundary, along x==N*/
For i In {0: N-2}
    Point(j) = { N, i + 0.5 + h/2.0, 0, cLen};                          //Lower tube, upper boundary node
    Point(j+1) = { N - 0.5 + circToTube, i + 0.5 + h/2.0, 0, cLen};     //Lower tube meets circle
    Point(j+2) = { N - 0.5 + h/2.0, i + 0.5 + circToTube, 0, cLen};     //Lower circle meets joining tube
    Point(j+3) = { N - 0.5 + h/2.0, (i+1) + 0.5 - circToTube, 0, cLen}; //Joining tube meets upper circle
    Point(j+4) = { N - 0.5 + circToTube, (i+1) + 0.5 - h/2.0, 0, cLen}; //Upper circle meets upper tube
    Point(j+5) = { N, (i+1) + 0.5 - h/2.0, 0, cLen};                    //Upper tube lower-right boundary node
    j = j+6;     //update tracker
EndFor
/*Manually add the final node in preparation for making the top boundary */
Point(j) = {N, N - 0.5 + h/2.0, 0, cLen};
j = j+1;
boundaryStarts += {j+1}; //the top boundary will start from the point with index j+1

/*Begin nodes along the top boundary, y==N*/
For i In {0 : N-1}
    Point(j) = { (N-i) - 0.5 + circToTube, N - 0.5 + h/2.0, 0, cLen};   //Left tube meets circle
    Point(j+1) = { (N-i) - 0.5 + h/2.0, N - 0.5 + circToTube, 0, cLen}; //Circle meets central (upward) tube on right
    Point(j+2) = { (N-i) - 0.5 + h/2.0, N, 0, cLen};                    //central tube upper-right boundary node
    Point(j+3) = { (N-i) - 0.5 - h/2.0, N, 0, cLen};                    //Central tube upper-left boundary node
    Point(j+4) = { (N-i) - 0.5 - h/2.0, N - 0.5 + circToTube, 0, cLen}; //central tube meets central circle on left
    Point(j+5) = { (N-i) - 0.5 - circToTube, N - 0.5 + h/2.0, 0, cLen}; //circle meets right tube
    j = j+6;    //update tracker
EndFor
/*Manually insert final node in preparation for making the left boundary */
Point(j) = {0, N - 0.5 + h/2.0, 0, cLen};
j = j+1;
boundaryStarts += {j+1}; //the left boundary starts at the node with index j+1

/*Begin nodes on along the left boundary, x==0*/
For i In {0 : N-2}
    Point(j) = { 0, (N-i) - 0.5 - h/2.0, 0, cLen};                      //Upper tube, lower boundary node
    Point(j+1) = { 0.5 - circToTube, (N-i) - 0.5 - h/2.0, 0, cLen};     //upper tube meets upper circle
    Point(j+2) = { 0.5 - h/2.0, (N-i) - 0.5 - circToTube, 0, cLen};     //upper circle meets joining tube
    Point(j+3) = { 0.5 - h/2.0, (N-i-1) - 0.5 + circToTube, 0, cLen};   //lower circle meets joining tube
    Point(j+4) = { 0.5 - circToTube, (N-i-1) - 0.5 + h/2.0, 0, cLen};   //lower circle meets lower tube
    Point(j+5) = { 0, (N-i-1) - 0.5 + h/2.0, 0, cLen};                //lower tube upper boundary node
    j = j+6;    //update tracker
EndFor

/*
We have now created all the boundary nodes that we require, however to actually draw the boundary we also need to define the centres of circles for the curved parts of the domain boundary.
As such, we do this now for all the cells, not just those "boundary cells".
Adopt the convention that the unit cell whose bottom left corner is at (i,k) has the node at it's circle's centre at index cellCircleStarts + k*N + i for i,k in {0:N-1}
So storage-wise, central-circle nodes are stored horizontally (along i) then the N-th entry moves to the second value of k, and begins the next "row".
*/
cellCircleStarts = newp;    //this is the next free index for points, which should be equal to j+1. This gives us the index at which we start to have nodes at the central circles of cells
cellCircles = {};           //this list stores all the circle-centre nodes, for access later.
For k In {0 : N-1}
	For i In {0 : N-1}
		cellCircles += {cellCircleStarts + k*N + i};                          //retain index of the node we are creating
		Point(cellCircleStarts + k*N + i) = {i + 0.5, k + 0.5, 0, cLen};      //create node a centre of circle
	EndFor
EndFor

/*
We now create the exterior boundary lines.
Given that we created these in order above, and tracked the beginning of each "quarter" of the boundary, this shouldn't be too hard.
We will however want some more lists to contain the IDs of the lines which are to be matched by periodicity later.
*/
bottomBoundaryPL = {};	rightBoundaryPL = {};	topBoundaryPL = {};	leftBoundaryPL = {};

/*Create the lines that make up the bottom boundary */
jj = 0;
For i In {0 : N-1}
    cellCircle = cellCircles(i);                                //centre of this cell's circle is just i, since we have k=0 and thus k*N + i = i.
    Line(newl) = {jj, jj+1};                                    //right tube of previous cell joins left this cell's circle (for i=0, left-boundary node joins circle of this cell)
    Circle(newl) = {jj+1, cellCircle, jj+2};                    //circle arc from left tube join to bottom-tube left-join
    Line(newl) = {jj+2, jj+3};                                  //bottom-tube left-circle join to bottom tube left-boundary
    bottomBoundaryPL += {newl};    Line(newl) = {jj+3, jj+4};   //bottom-tube left boundary to right boundary - should be periodic, hence it's index is saved prior to creation.
    Line(newl) = {jj+4, jj+5};                                  //bottom-tube right boundary to bottom-tube right circle join
    Circle(newl) = {jj+5, cellCircle, jj+6};                    //bottom-tube right circle join to right-tube circle join
    Line(newl) = {jj+6, jj+7};                                  //right-tube circle join to right-tube cell boundary
    jj = jj+7;      //update tracker
EndFor

/*Create lines that make up the right boundary */
For k In {0 : N-2}
    lowCircCentre = cellCircles(k*N + (N-1));                   //centre of the lower cell's circle for this portion of the boundary
    upCircCentre = cellCircles((k+1)*N + (N-1));                //centre of the upper cell's circle for this portion of the boundary
    rightBoundaryPL += {newl};    Line(newl) = {jj, jj+1};      //lower-tube lower boundary node to lower-tube upper boundary node
    Line(newl) = {jj+1, jj+2};                                  //lower-tube upper boundary node to lower-circle right join
    Circle(newl) = {jj+2, lowCircCentre, jj+3};                 //lower-circle right join to lower-circle central tube join
    Line(newl) = {jj+3, jj+4};                                  //central tube right edge joining upper and lower circles
    Circle(newl) = {jj+4, upCircCentre, jj+5};                  //central tube joins upper-circle to upper-circle joins upper tube
    Line(newl) = {jj+5, jj+6};                                  //upper-circle joins upper tube to upper-tube lower boundary node
    jj = jj+6;      //update tracker
EndFor
/*We don't add the final line on the right boundary in this loop, so we need to add it manually. It's also part of the periodic boundary too! */
rightBoundaryPL += {newl};    Line(newl) = {jj, jj+1};          //edge of the right-tube in cell (N-1,N-1)
jj = jj+1;

/*Create lines for the top boundary */
For i In {0 : N-1}
    cellCircle = cellCircles((N-1)*N + N-1 -i);                 //this cell's circle centre. k=N-1, and we are going BACKWARDS from (N-1,N-1) through (N-1,0)
    Line(newl) = {jj, jj+1};                                    //right-edge of previous cell's left-tube joins circle of this cell
    Circle(newl) = {jj+1, cellCircle, jj+2};                    //right-tube circle join to upper-tube circle join
    Line(newl) = {jj+2, jj+3};                                  //upper tube right edge
    topBoundaryPL += {newl};    Line(newl) = {jj+3, jj+4};      //upper tube top boundary - part of the periodic edge
    Line(newl) = {jj+4, jj+5};                                  //upper tube left edge
    Circle(newl) = {jj+5, cellCircle, jj+6};                    //upper-tube left join to left-tube circle join
    jj = jj+6;      //update tracker
EndFor
/*We don't add the final line on the top boundary in this loop, so manually add it */
Line(newl) = {jj, jj+1};                                        //top edge of the left-most tube in cell (0, N-1)
jj = jj+1;

/*Create lines for the left boundary */
For k In {0: N-2}
    lowCircCentre = cellCircles((N-2-k)*N);                     //centre of lower cell's circle for this portion of the boundary [cell (N-2-k, 0)]
    upCircCentre = cellCircles((N-1-k)*N);                      //centre of upper cell's circle for this portion of the boundary [cell (N-1-k, 0)]
    leftBoundaryPL += {newl};   Line(newl) = {jj, jj+1};        //lower-tube boundary from previous cell links to upper-tube boundary of this cell, and is part of the periodic boundary
    Line(newl) = {jj+1, jj+2};                                  //upper-tube boundary node to upper-tube upper-circle join
    Circle(newl) = {jj+2, upCircCentre, jj+3};                  //upper-tube upper-circle join to upper-circle central-tube join
    Line(newl) = {jj+3, jj+4};                                  //central tube left edge
    Circle(newl) = {jj+4, lowCircCentre, jj+5};                 //central-tube lower-circle join to lower-circle lower-tube join
    Line(newl) = {jj+5, jj+6};                                  //lower tube top edge
    jj = jj+6;      //update tracker
EndFor
/*We don't add the final line on the left boundary in this loop, so manually add it. It's also part of the periodic boundary! */
leftBoundaryPL += {newl};   Line(newl) = {jj, 0};            //left boundary of left-tube in cell (0,0)

/*Now we can define the exterior boundary loop */
bllID = newl - 1;					//"Boundary Lines Last ID": 1:bllID are boundary lines that need to be looped
bLoopID = newl;						//Boundary loop ID for later reference
Line Loop(bLoopID) = {1 : bllID};	//Boundary loop
Printf("Create exterior boundary loop, ID = %.0f. Beginning interior boundaries.", bLoopID);

/*
Next, we create the domain "holes", IE the interior boundaries.
In this approach, we can be crafty though - we can create the "hole" at the intersection of the cells (0,0), (0,1), (1,1) and (1,0), then copy and translate this as needed to make the other "holes".
*/
holeStartIndex = newp;			//all points created from here on out are for use with the holes
/* First hole is centred on the point (x,y) = (1,1), corresponding to the intersection of the 4 cells with IDs listed above.
We need the 4 circle centres to create the arcs needed for the interior boundary */
blCirc = cellCircles(0);	brCirc = cellCircles(1);
tlCirc = cellCircles(N);	trCirc = cellCircles(N+1);
/*Now let's create the interior boundary of one hole, centred (1,1).
First, make the nodes.
Start from the BL circle meets bottom tube node, and proceed counterclockwise */
Point(newp) = {0.5 + circToTube, 0.5 + h/2.0, 0, cLen};     //BL circle meets bottom tube
Point(newp) = {1.5 - circToTube, 0.5 + h/2.0, 0, cLen};     //bottom tube meets BR circle
Point(newp) = {1.5 - h/2.0, 0.5 + circToTube, 0, cLen};     //BR circle meets right tube
Point(newp) = {1.5 - h/2.0, 1.5 - circToTube, 0, cLen};     //right tube meets TR circle
Point(newp) = {1.5 - circToTube, 1.5 - h/2.0, 0, cLen};     //TR circle meets top tube
Point(newp) = {0.5 + circToTube, 1.5 - h/2.0, 0, cLen};     //top tube meets TL circle
Point(newp) = {0.5 + h/2.0, 1.5 - circToTube, 0, cLen};     //TL circle meets left tube
Point(newp) = {0.5 + h/2.0, 0.5 + circToTube, 0, cLen};     //left tube meets BL circle
/*Now create the lines that make up the interior boundary of one hole */
hLS = newl;						                           //ID at which the hole lines start ("hole line start")
Line(newl) = {holeStartIndex, holeStartIndex+1};                //bottom tube edge
Circle(newl) = {holeStartIndex+1, brCirc ,holeStartIndex+2};    //BR circle arc
Line(newl) = {holeStartIndex+2, holeStartIndex+3};              //right tube edge
Circle(newl) = {holeStartIndex+3, trCirc, holeStartIndex+4};    //TR circle arc
Line(newl) = {holeStartIndex+4, holeStartIndex+5};              //top tube edge
Circle(newl) = {holeStartIndex+5, tlCirc, holeStartIndex+6};    //TL circle arc
Line(newl) = {holeStartIndex+6, holeStartIndex+7};              //left tube edge
Circle(newl) = {holeStartIndex+7, blCirc, holeStartIndex};      //BL circle arc
/*Now turn the hole into a line loop and create an ID for that loop, so we can specify it as an interior boundary.
Also record it's ID, so that we can specify the holes when we create our domain. */
hLE = newl - 1;					                        //ID at which the hole lines end ("hole line end")
hLoopID = newl;					                        //ID of the hole which we will make copies of to complete the domain
hLoopList = {hLS : hLE};
Line Loop(hLoopID) = hLoopList[];		                //Create hole line loop

/*Create lists to store all the "loops" that make up the boundaries of the domain.
Index 0 is the ID of the exterior loop, indices 1:end are the IDs of the interior boundaries (holes) */
domainEdgeIDs = {}; domainEdgeIDs += {bLoopID};         //first entry is be the exterior boundary ID
/*Create a sufficient number of copies of the holes and place them into the domain.
We checked right at the start whether N<2, so we've already warned the user that N<1 is not to be used! */
If (N==2)
	/*If N==2 there is only one hole, and it is where the "reference" hole just created is. Thus append to a list and exit.*/
    Printf("N==%.0f: insert reference hole ID and finish.", N);
	domainEdgeIDs += {hLoopID};
Else
	/*In general, we have (N-1)^2 holes and they need to be appropriately centred and translated.
	We also need to keep track of their IDs so we can list all the holes. */
    Printf("N==%.0f: creating and translating additional holes.", N);
	domainEdgeIDs += {hLoopID};                //add the original hole we made, since this is one of the holes in the domain anyway
	For i In {0 : N-2}
		For j In {0 : N-2}
			If ( !((i==0) && (j==0)) )	      //avoid creating a copy of the original hole on top of itself
				IDstart = newl;
				Translate {i, j, 0} { Duplicata{ Line{hLoopList[]}; } } //duplicate the list of edges in hLoopList, then translate them by the vector (i,j,0) provided
				hold = newl;    Line Loop(hold) = {IDstart : hold-1};   //create a new loop to serve as the boundary, for the translated copy of the hole just created
				domainEdgeIDs += {hold};
			EndIf
		EndFor
	EndFor
EndIf
/*We can now create the plane surface for our domain. */
domainID = news;                                //the (surface) ID of our domain
Plane Surface(domainID) = domainEdgeIDs[];      //create this surface

/*Finally, we need to match the periodic boundaries with each other.
The exterior boundary edges have IDs 1:bllID.
We have 4 lists containing the relevant edges that are to be matched by periodicity, so it's just a case of matching them up.
Should run some checks first as to whether the mesh is correctly setup: */
If ( #topBoundaryPL[]==#bottomBoundaryPL[] )
	lengthTB = #bottomBoundaryPL[];
Else
    Error("Found uneven number of top and bottom edges to match, %.0f vs %.0f - ABORT", #topBoundaryPL[], #bottomBoundaryPL[]);
    Abort;
EndIf
If ( #rightBoundaryPL[]==#leftBoundaryPL[] )
	lengthRL = #rightBoundaryPL[];
Else
    Error("Found uneven number of left and right edges to match, %.0f vs %.0f - ABORT", #leftBoundaryPL[], #rightBoundaryPL[]);
    Abort;
EndIf
/*If we got to here, then there is an equal number of lines/edges to be made periodic and matched.
We now do this by looping through our lists. */
For i In {0 : lengthTB-1 }
	/*Enslave top boundary to bottom boundary, recalling that the lines need to be orientated correctly as they "go" in opposite directions.
    This is the reason for the - sign in the command below, the top edges were created "moving" from right -> left, but the bottom edges were creating moving left -> right. */
    Printf("Matching line %.0f to line %.0f (periodicity top-bottom)", topBoundaryPL(lengthTB-1-i), bottomBoundaryPL(i));
	Periodic Line{topBoundaryPL(lengthTB-1-i)} = {-bottomBoundaryPL(i)};
EndFor
For i In {0 : lengthRL-1 }
	/*Enslave left boundary to right boundary, recalling that the lines need to be orientated correctly as they "go" in opposite directions.
    This is the reason for the - sign in the command below, the left edges were created "moving" from top -> bottom, but the right edges were creating moving bottom -> top. */
    Printf("Matching line %.0f to line %.0f (periodicity left-right)", leftBoundaryPL(lengthRL-1-i), rightBoundaryPL(i));
	Periodic Line{leftBoundaryPL(lengthRL-1-i)} = {-rightBoundaryPL(i)};
EndFor

Printf("Meshing complete");
