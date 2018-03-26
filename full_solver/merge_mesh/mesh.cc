/*
 * Moving boundary being added to the daes
 *
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/sparse_ilu.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;
using namespace numbers;
namespace data
{
//    const double left = 0;
//    const double right = 4;
//    const double bottom = 0;
//    const double top = 4;
    const unsigned int dim =2;
    
}
using namespace data;

template <int dim>
void print_mesh_info(const Triangulation<dim> &tria,
                     const std::string        &filename)
{
    std::cout << "Mesh info:" << std::endl
    << " dimension: " << dim << std::endl
    << " no. of cells: " << tria.n_active_cells() << std::endl
    << " no. of vertices: " << tria.n_vertices() << std::endl;
    {
        std::map<unsigned int, unsigned int> boundary_count;
        typename Triangulation<dim>::active_cell_iterator
        cell = tria.begin_active(),
        endc = tria.end();
        for (; cell!=endc; ++cell)
        {
            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
                if (cell->face(face)->at_boundary())
                    boundary_count[cell->face(face)->boundary_id()]++;
            }
        }
        std::cout << " boundary indicators: ";
        for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
             it!=boundary_count.end();
             ++it)
        {
            std::cout << it->first << "(" << it->second << " times) ";
        }
        std::cout << std::endl;
    }
    std::ofstream out (filename.c_str());
    GridOut grid_out;
    grid_out.write_eps (tria, out);
    std::cout << " written to " << filename
    << std::endl
    << std::endl;
}


void my_grid ()
{
    Triangulation<2> tria1;
//    GridGenerator::hyper_cube_with_cylindrical_hole (tria1, 0.25, 1.0);
    Triangulation<2> tria2;
    std::vector< unsigned int > repetitions(2);
    repetitions[0]=2;
    repetitions[1]=2;
    GridGenerator::subdivided_hyper_rectangle (tria1, repetitions,
                                               Point<2>(0.0,0.0),
                                               Point<2>(2,2));
    print_mesh_info(tria1, "grid-1.eps");

    std::vector<Point<2>> vertex_points;
    unsigned int p = 0;
    
    for (typename Triangulation<dim>::active_cell_iterator
         cell = tria1.begin_active();
         cell != tria1.end(); ++cell)
    {
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
            if (cell->face(f)->center()[dim-1] == 2.0)
            {
                cell->face(f)->set_all_boundary_ids(1);
                Point<2> points = cell->face(f)->vertex(0);
                vertex_points.push_back (points);
                ++p;
            }
        }
    }
    vertex_points.push_back (Point<2> (2.,2.));
    vertex_points.push_back (Point<2> (0.,4.));
    vertex_points.push_back (Point<2> (1.,4.));
    vertex_points.push_back (Point<2> (2.,4.));
//    std::cout << vertex_points.size() << std::endl;
    
    static const int cell_vertices[][GeometryInfo<dim>::vertices_per_cell]
    = {{0, 1, 3, 4},
        {1, 2, 4, 5}
    };
    const unsigned int
    n_cells = sizeof(cell_vertices) / sizeof(cell_vertices[0]);
    std::vector<CellData<dim> > cells (n_cells, CellData<dim>());
    for (unsigned int i=0; i<n_cells; ++i)
    {
        for (unsigned int j=0;
             j<GeometryInfo<dim>::vertices_per_cell;
             ++j)
            cells[i].vertices[j] = cell_vertices[i][j];
        cells[i].material_id = 0;
    }
    
    tria2.create_triangulation (vertex_points,
                                cells,
                                SubCellData());
    for (typename Triangulation<dim>::active_cell_iterator
         cell = tria2.begin_active();
         cell != tria2.end(); ++cell)
    {
        cell->set_refine_flag(RefinementCase<dim>::cut_y);
    }
    print_mesh_info(tria2, "grid-2.eps");


//
    Triangulation<2> triangulation;
    GridGenerator::merge_triangulations (tria1, tria2, triangulation);
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
    {
        for (unsigned int i=0; i<dim*2; ++i) // cell has 4 vertices each in 2d.
        {
            if (cell->vertex(i)[1] > 2.0)
                cell->set_refine_flag(RefinementCase<dim>::cut_y);
        }
    }
    triangulation.execute_coarsening_and_refinement ();
    triangulation.refine_global (2);
    print_mesh_info(triangulation, "final_grid.eps");

    
}

int main ()
{
    my_grid ();
    return 0;
}
