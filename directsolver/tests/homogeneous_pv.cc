/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2008 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 
 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2008
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

namespace Step22
{
    using namespace dealii;
    
    namespace data
    {
        const double rho_B = 0.0;
        const double eta = 1.0;
        const double top = 2.0;
        const double bottom = 0.0;
        const double p_top = 0.0;
        const double p_bottom = 10000.0;
        const int dimension = 2;
        
    }
    

    
    
    template <int dim>
    class StokesProblem
    {
    public:
        StokesProblem (const unsigned int degree);
        void run ();
        
    private:
        void setup_dofs ();
        void assemble_system ();
        void solve ();
        void output_results (const unsigned int refinement_cycle) const;
        void refine_mesh ();
        
        const unsigned int   degree;
        
        Triangulation<dim>   triangulation;
        FESystem<dim>        fe;
        DoFHandler<dim>      dof_handler;
        
        
        BlockSparsityPattern      sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        
        BlockVector<double> solution;
        BlockVector<double> system_rhs;
        

    };
    
    
    template <int dim>
    StokesProblem<dim>::StokesProblem (const unsigned int degree)
    :
    degree (degree),
    triangulation (Triangulation<dim>::maximum_smoothing),
    fe (FE_Q<dim>(degree+1), dim,
        FE_Q<dim>(degree), 1),
    dof_handler (triangulation)
    {}
    
    
    
    
    template <int dim>
    void StokesProblem<dim>::setup_dofs ()
    {

        system_matrix.clear ();
        
        dof_handler.distribute_dofs (fe);
        DoFRenumbering::Cuthill_McKee (dof_handler);
        
        std::vector<unsigned int> block_component (dim+1,0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise (dof_handler, block_component);

        
        
        
        std::vector<types::global_dof_index> dofs_per_block (2);
        DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0],
        n_p = dofs_per_block[1];
        
        std::cout << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << " (" << n_u << '+' << n_p << ')'
        << std::endl;
        
        {
            BlockDynamicSparsityPattern dsp (2,2);
            
            dsp.block(0,0).reinit (n_u, n_u);
            dsp.block(1,0).reinit (n_p, n_u);
            dsp.block(0,1).reinit (n_u, n_p);
            dsp.block(1,1).reinit (n_p, n_p);
            
            dsp.collect_sizes();
            
            DoFTools::make_sparsity_pattern (dof_handler, dsp);
            sparsity_pattern.copy_from (dsp);
        }
        
        system_matrix.reinit (sparsity_pattern);
        
        solution.reinit (2);
        solution.block(0).reinit (n_u);
        solution.block(1).reinit (n_p);
        solution.collect_sizes ();
        
        system_rhs.reinit (2);
        system_rhs.block(0).reinit (n_u);
        system_rhs.block(1).reinit (n_p);
        system_rhs.collect_sizes ();
    }
    
    

    template <int dim>
    void StokesProblem<dim>::assemble_system ()
    {
        system_matrix=0;
        system_rhs=0;
        
        QGauss<dim>   quadrature_formula(degree+2);
        QGauss<dim-1> face_quadrature_formula(degree+2);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 update_gradients);
        
        FEFaceValues<dim> fe_face_values ( fe, face_quadrature_formula,
                                          update_values |
                                          update_normal_vectors |
                                          update_quadrature_points |
                                          update_JxW_values
                                          );
        
        const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
        
        const unsigned int   n_q_points      = quadrature_formula.size();
        
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        

        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        std::vector<Tensor<1,dim> >          phi_u       (dofs_per_cell);
        std::vector<Tensor<2,dim> >          grad_phi_u (dofs_per_cell);
        std::vector<double>                  div_phi_u   (dofs_per_cell);
        std::vector<double>                  phi_p       (dofs_per_cell);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            local_matrix = 0;
            local_rhs = 0;
            
            
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                    phi_u[k]         = fe_values[velocities].value (k, q);
                    grad_phi_u[k]    = fe_values[velocities].gradient (k, q);
                    div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                    phi_p[k]         = fe_values[pressure].value (k, q);
                }
                
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                        local_matrix(i,j) += (2 * data::eta *
                                              scalar_product
                                              (grad_phi_u[i] ,grad_phi_u[j])
                                              - div_phi_u[i] * phi_p[j]
                                              - phi_p[i] * div_phi_u[j] )
                        * fe_values.JxW(q);
                        
                    }
                 

                }
                
            }

            
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                system_matrix.add (local_dof_indices[i],
                                   local_dof_indices[j],
                                   local_matrix(i,j));
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              system_rhs(local_dof_indices[i]) += local_rhs(i);
        }
        
        
    }


    
    template <int dim>
    void StokesProblem<dim>::solve ()
    {

        SparseDirectUMFPACK  A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult (solution, system_rhs);

    }

    
    
    
    
    template <int dim>
    void
    StokesProblem<dim>::output_results (const unsigned int refinement_cycle)  const
    {
        std::vector<std::string> solution_names (dim, "velocity");
        solution_names.push_back ("pressure");
        
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation
        (dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation
        .push_back (DataComponentInterpretation::component_is_scalar);
        
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (solution, solution_names,
                                  DataOut<dim>::type_dof_data,
                                  data_component_interpretation);
        data_out.build_patches ();
        
        std::ostringstream filename;
        filename << "solution-"
        << Utilities::int_to_string (refinement_cycle, 2)
        << ".vtk";
        
        std::ofstream output (filename.str().c_str());
        data_out.write_vtk (output);
    }
    
    
    
    template <int dim>
    void StokesProblem<dim>::run ()
    {
        {
            std::vector<unsigned int> subdivisions (dim, 1);
            subdivisions[0] = 4;
            
            const Point<dim> bottom_left = (dim == 2 ?
                                            Point<dim>(0,data::bottom) :
                                            Point<dim>(-2,0,-1));
            const Point<dim> top_right   = (dim == 2 ?
                                            Point<dim>(1,data::top) :
                                            Point<dim>(2,1,0));
            
            GridGenerator::subdivided_hyper_rectangle (triangulation,
                                                       subdivisions,
                                                       bottom_left,
                                                       top_right);
        }

        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->center()[dim-1] == data::top)
                    cell->face(f)->set_all_boundary_ids(1);
                else if (cell->face(f)->center()[dim-1] == data::bottom)
                    cell->face(f)->set_all_boundary_ids(2);

        
        
        triangulation.refine_global (5);
        

            setup_dofs ();
            
            std::cout << "   Assembling..." << std::endl << std::flush;
            assemble_system ();
            
            std::cout << "   Solving..." << std::flush;
            solve ();
            
            output_results (1);
            
            std::cout << std::endl;
    }
}



int main ()
{
    try
    {
        using namespace dealii;
        using namespace Step22;
        
        StokesProblem<data::dimension> flow_problem(1);
        flow_problem.run ();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    
    return 0;
}
