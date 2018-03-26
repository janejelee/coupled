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

template <int dim>
void print_mesh_info(const Triangulation<dim> &tria,
                     const std::string        &filename)
{
    std::cout << "Mesh info:" << std::endl
    << " dimension: " << dim << std::endl
    << " no. of cells: " << tria.n_active_cells() << std::endl;
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


void grid_2 ()
{
    Triangulation<2> tria1;
    GridGenerator::hyper_cube_with_cylindrical_hole (tria1, 0.25, 1.0);
    Triangulation<2> tria2;
    std::vector< unsigned int > repetitions(2);
    repetitions[0]=3;
    repetitions[1]=2;
    GridGenerator::subdivided_hyper_rectangle (tria2, repetitions,
                                               Point<2>(1.0,-1.0),
                                               Point<2>(4.0,1.0));
    Triangulation<2> triangulation;
    GridGenerator::merge_triangulations (tria1, tria2, triangulation);
    print_mesh_info(triangulation, "grid-2.eps");
}




int main ()
{
    grid_2 ();
    return 0;
}
