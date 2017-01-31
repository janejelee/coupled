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
        const double rho_B = 1.0;
        const double eta = 1.0;
        const double top = 1.0;
        const double bottom = 0.0;
        const double p_top = 10.0;
        const double p_bottom = 100.0;
        const int dimension = 2;
        
    };
    
    
    template <int dim>
    struct InnerPreconditioner;
    
    template <>
    struct InnerPreconditioner<2>
    {
        typedef SparseDirectUMFPACK type;
    };
    
    template <>
    struct InnerPreconditioner<3>
    {
        typedef SparseILU<double> type;
    };
    
    
    
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
        
        ConstraintMatrix     constraints;
        
        BlockSparsityPattern      sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        
        BlockVector<double> solution;
        BlockVector<double> system_rhs;
        
        std_cxx11::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;
    };
    
  ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    


    
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////
    

    template <class Matrix, class Preconditioner>
    class InverseMatrix : public Subscriptor
    {
    public:
        InverseMatrix (const Matrix         &m,
                       const Preconditioner &preconditioner);
        
        void vmult (Vector<double>       &dst,
                    const Vector<double> &src) const;
        
    private:
        const SmartPointer<const Matrix> matrix;
        const SmartPointer<const Preconditioner> preconditioner;
    };
    
    
    template <class Matrix, class Preconditioner>
    InverseMatrix<Matrix,Preconditioner>::InverseMatrix (const Matrix &m,
                                                         const Preconditioner &preconditioner)
    :
    matrix (&m),
    preconditioner (&preconditioner)
    {}
    
    
    
    template <class Matrix, class Preconditioner>
    void InverseMatrix<Matrix,Preconditioner>::vmult (Vector<double>       &dst,
                                                      const Vector<double> &src) const
    {
        SolverControl solver_control (src.size(), 1e-6*src.l2_norm());
        SolverCG<>    cg (solver_control);
        
        dst = 0;
        
        cg.solve (*matrix, dst, src, *preconditioner);
    }
    
    
    
    template <class Preconditioner>
    class SchurComplement : public Subscriptor
    {
    public:
        SchurComplement (const BlockSparseMatrix<double> &system_matrix,
                         const InverseMatrix<SparseMatrix<double>, Preconditioner> &A_inverse);
        
        void vmult (Vector<double>       &dst,
                    const Vector<double> &src) const;
        
    private:
        const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
        const SmartPointer<const InverseMatrix<SparseMatrix<double>, Preconditioner> > A_inverse;
        
        mutable Vector<double> tmp1, tmp2;
    };
    
    
    
    template <class Preconditioner>
    SchurComplement<Preconditioner>::
    SchurComplement (const BlockSparseMatrix<double> &system_matrix,
                     const InverseMatrix<SparseMatrix<double>,Preconditioner> &A_inverse)
    :
    system_matrix (&system_matrix),
    A_inverse (&A_inverse),
    tmp1 (system_matrix.block(0,0).m()),
    tmp2 (system_matrix.block(0,0).m())
    {}
    
    
    template <class Preconditioner>
    void SchurComplement<Preconditioner>::vmult (Vector<double>       &dst,
                                                 const Vector<double> &src) const
    {
        system_matrix->block(0,1).vmult (tmp1, src);
        A_inverse->vmult (tmp2, tmp1);
        system_matrix->block(1,0).vmult (dst, tmp2);
    }
    
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    
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
        A_preconditioner.reset ();
        system_matrix.clear ();
        
        dof_handler.distribute_dofs (fe);
        DoFRenumbering::Cuthill_McKee (dof_handler);
        
        std::vector<unsigned int> block_component (dim+1,0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise (dof_handler, block_component);
        
        {
            constraints.clear ();
            
            FEValuesExtractors::Vector velocities(0);
            DoFTools::make_hanging_node_constraints (dof_handler,
                                                     constraints);
            
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (0);
            VectorTools::compute_no_normal_flux_constraints (dof_handler, 0,
                                                             no_normal_flux_boundaries,
                                                             constraints);

        }
        
        constraints.close ();
        
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
            
            DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
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
    
  /////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    
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
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        

        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        std::vector<Tensor<1,dim> >          phi_u       (dofs_per_cell);
        std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);
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
                    symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                    grad_phi_u[k]    = fe_values[velocities].gradient (k, q);
                    div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                    phi_p[k]         = fe_values[pressure].value (k, q);
                }
                
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j=0; j<=i; ++j)
                    {
                        local_matrix(i,j) += (2 *
                                              scalar_product
                                              (grad_phi_u[i] ,grad_phi_u[j])
                                              - div_phi_u[i] * phi_p[j]
                                              - phi_p[i] * div_phi_u[j]
                                              + phi_p[i] * phi_p[j])
                        * fe_values.JxW(q);
                        
                    }
                 
                    const Point<dim> gravity = ( (dim == 2) ? (Point<dim> (0,1)) :
                                                 (Point<dim> (0,0,1)) );

                    
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                        local_rhs(i) += (data::rho_B *
                                         gravity * phi_u[i] )*
                                         fe_values.JxW(q);
                }
                
            }
            
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                    local_matrix(i,j) = local_matrix(j,i);
            
            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                if (cell->face(face_number)->at_boundary()
                    &&
                    (cell->face(face_number)->boundary_id() == 1))
                {
                    fe_face_values.reinit (cell, face_number);
                    
                   
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                        const double stress_value
                        = data::p_top;
                        
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                            local_rhs(i) += (stress_value *
                                            fe_face_values.shape_value(i,q_point) *
                                            fe_face_values.JxW(q_point));
                    }
                }
                else if (cell->face(face_number)->at_boundary()
                         &&
                         (cell->face(face_number)->boundary_id() == 2))
                {
                    fe_face_values.reinit (cell, face_number);
                    
                    
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                        const double stress_value
                        = data::p_bottom;
                        
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                            local_rhs(i) += (stress_value *
                                            fe_face_values.shape_value(i,q_point) *
                                            fe_face_values.JxW(q_point));
                    }
                    
                    
                    
                }

            
            cell->get_dof_indices (local_dof_indices);
            constraints.distribute_local_to_global (local_matrix, local_rhs,
                                                    local_dof_indices,
                                                    system_matrix, system_rhs);
        }
        
        std::cout << "   Computing preconditioner..." << std::endl << std::flush;
        
        A_preconditioner
        = std_cxx11::shared_ptr<typename InnerPreconditioner<dim>::type>(new typename InnerPreconditioner<dim>::type());
        A_preconditioner->initialize (system_matrix.block(0,0),
                                      typename InnerPreconditioner<dim>::type::AdditionalData());
        
    }
    
    /////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    
    template <int dim>
    void StokesProblem<dim>::solve ()
    {
        const InverseMatrix<SparseMatrix<double>,
        typename InnerPreconditioner<dim>::type>
        A_inverse (system_matrix.block(0,0), *A_preconditioner);
        Vector<double> tmp (solution.block(0).size());
        
        {
            Vector<double> schur_rhs (solution.block(1).size());
            A_inverse.vmult (tmp, system_rhs.block(0));
            system_matrix.block(1,0).vmult (schur_rhs, tmp);
            schur_rhs -= system_rhs.block(1);
            
            SchurComplement<typename InnerPreconditioner<dim>::type>
            schur_complement (system_matrix, A_inverse);
            
            SolverControl solver_control (solution.block(1).size(),
                                          1e-6*schur_rhs.l2_norm());
            SolverCG<>    cg (solver_control);
            
            SparseILU<double> preconditioner;
            preconditioner.initialize (system_matrix.block(1,1),
                                       SparseILU<double>::AdditionalData());
            
            InverseMatrix<SparseMatrix<double>,SparseILU<double> >
            m_inverse (system_matrix.block(1,1), preconditioner);
            
            cg.solve (schur_complement, solution.block(1), schur_rhs,
                      m_inverse);
            
            constraints.distribute (solution);
            
            std::cout << "  "
            << solver_control.last_step()
            << " outer CG Schur complement iterations for pressure"
            << std::endl;
        }
        
        {
            system_matrix.block(0,1).vmult (tmp, solution.block(1));
            tmp *= -1;
            tmp += system_rhs.block(0);
            
            A_inverse.vmult (solution.block(0), tmp);
            
            constraints.distribute (solution);
        }
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
    void
    StokesProblem<dim>::refine_mesh ()
    {
        Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
        
        FEValuesExtractors::Scalar pressure(dim);
        KellyErrorEstimator<dim>::estimate (dof_handler,
                                            QGauss<dim-1>(degree+1),
                                            typename FunctionMap<dim>::type(),
                                            solution,
                                            estimated_error_per_cell,
                                            fe.component_mask(pressure));
        
        GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                         estimated_error_per_cell,
                                                         0.3, 0.0);
        triangulation.execute_coarsening_and_refinement ();
    }
    
    
    
    template <int dim>
    void StokesProblem<dim>::run ()
    {
        {
            std::vector<unsigned int> subdivisions (dim, 1);
            subdivisions[0] = 4;
            
            const Point<dim> bottom_left = (dim == 2 ?
                                            Point<dim>(-2,data::bottom) :
                                            Point<dim>(-2,0,-1));
            const Point<dim> top_right   = (dim == 2 ?
                                            Point<dim>(2,data::top) :
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

        
        
        triangulation.refine_global (6);
        
        /* for (unsigned int refinement_cycle = 0; refinement_cycle<2;
             ++refinement_cycle)
        {
            std::cout << "Refinement cycle " << refinement_cycle << std::endl;
            
            /*
            if (refinement_cycle > 0)
                refine_mesh ();
             */
            
            setup_dofs ();
            
            std::cout << "   Assembling..." << std::endl << std::flush;
            assemble_system ();
            
            std::cout << "   Solving..." << std::flush;
            solve ();
            
            output_results (1);
            
            std::cout << std::endl;
        /* } */
    
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
