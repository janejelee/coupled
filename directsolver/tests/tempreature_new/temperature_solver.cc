
/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2015 by the deal.II authors
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
 * Author: Wolfgang Bangerth and Ralf Hartmann, University of Heidelberg, 2000
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>

#include <typeinfo>
#include <fstream>
#include <iostream>

namespace Step7
{
    using namespace dealii;
    using namespace numbers;
    
    namespace data
    {
        const int no_cycles = 3;
        const int dimension = 2;
        int degree;
        const double top = 1.0;
        const double bottom = 0.0;
        const double right = PI;
        
        const double timestep = 0.0000001;
        const double final_time = 11*timestep;
        const double error_time = 13;
        
        
        const int global_refinement_level = 2;
        
        
        
    }
    
    template <int dim>
    class SolutionBase
    {
    protected:
        static const unsigned int n_source_centers = 3;
        static const Point<dim>   source_centers[n_source_centers];
        static const double       width;
    };
    
    
    template <>
    const Point<1>
    SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers]
    = { Point<1>(-1.0 / 3.0),
        Point<1>(0.0),
        Point<1>(+1.0 / 3.0)
    };
    
    template <>
    const Point<2>
    SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers]
    = { Point<2>(-0.5, +0.5),
        Point<2>(-0.5, -0.5),
        Point<2>(+0.5, -0.5)
    };
    
    template <int dim>
    const double SolutionBase<dim>::width = 1./8.;
    
    
    
    template <int dim>
    class Solution : public Function<dim>,
    protected SolutionBase<dim>
    {
    public:
        Solution () : Function<dim>() {}
        
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
        
        virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                        const unsigned int  component = 0) const;
    };
    
    
    template <int dim>
    double Solution<dim>::value (const Point<dim>   &p,
                                 const unsigned int) const
    {
        double return_value = 0;
        for (unsigned int i=0; i<this->n_source_centers; ++i)
        {
            const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
            return_value += std::exp(-x_minus_xi.norm_square() /
                                     (this->width * this->width));
        }
        
        return return_value;
    }
    
    
    template <int dim>
    Tensor<1,dim> Solution<dim>::gradient (const Point<dim>   &p,
                                           const unsigned int) const
    {
        Tensor<1,dim> return_value;
        
        for (unsigned int i=0; i<this->n_source_centers; ++i)
        {
            const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
            
            return_value += (-2 / (this->width * this->width) *
                             std::exp(-x_minus_xi.norm_square() /
                                      (this->width * this->width)) *
                             x_minus_xi);
        }
        
        return return_value;
    }
    
    
    
    template <int dim>
    class RightHandSide : public Function<dim>,
    protected SolutionBase<dim>
    {
    public:
        RightHandSide () : Function<dim>() {}
        
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    
    template <int dim>
    double RightHandSide<dim>::value (const Point<dim>   &p,
                                      const unsigned int) const
    {
        double return_value = 0;
        for (unsigned int i=0; i<this->n_source_centers; ++i)
        {
            const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
            
            return_value += ((2*dim - 4*x_minus_xi.norm_square()/
                              (this->width * this->width)) /
                             (this->width * this->width) *
                             std::exp(-x_minus_xi.norm_square() /
                                      (this->width * this->width)));
            return_value += std::exp(-x_minus_xi.norm_square() /
                                     (this->width * this->width));
        }
        
        return return_value;
    }
    
    
    
    template <int dim>
    class HelmholtzProblem
    {
    public:
        enum RefinementMode
        {
            global_refinement, adaptive_refinement
        };
        
        HelmholtzProblem (const FiniteElement<dim> &fe,
                          const RefinementMode      refinement_mode);
        
        ~HelmholtzProblem ();
        
        void run ();
        
    private:
        void setup_system ();
        void assemble_system ();
        void assemble_rhs ();
        void solve ();
        void refine_grid ();
        void process_solution (const unsigned int cycle);
        void table ();
        
        Triangulation<dim>                      triangulation;
        DoFHandler<dim>                         dof_handler;
        
        SmartPointer<const FiniteElement<dim> > fe;
        
        ConstraintMatrix                        hanging_node_constraints;
        
        SparsityPattern                         sparsity_pattern;
        SparseMatrix<double>                    system_matrix;
        
        Vector<double>                          solution;
        Vector<double>                          system_rhs;
        
        const RefinementMode                    refinement_mode;
        
        ConvergenceTable                        convergence_table;
    };
    
    
    
    
    template <int dim>
    HelmholtzProblem<dim>::HelmholtzProblem (const FiniteElement<dim> &fe,
                                             const RefinementMode refinement_mode) :
    dof_handler (triangulation),
    fe (&fe),
    refinement_mode (refinement_mode)
    {}
    
    
    
    template <int dim>
    HelmholtzProblem<dim>::~HelmholtzProblem ()
    {
        dof_handler.clear ();
    }
    
    
    
    template <int dim>
    void HelmholtzProblem<dim>::setup_system ()
    {
        dof_handler.distribute_dofs (*fe);
        DoFRenumbering::Cuthill_McKee (dof_handler);
        
        hanging_node_constraints.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler,
                                                 hanging_node_constraints);
        hanging_node_constraints.close ();
        
        DynamicSparsityPattern dsp (dof_handler.n_dofs(), dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern (dof_handler, dsp);
        hanging_node_constraints.condense (dsp);
        sparsity_pattern.copy_from (dsp);
        
        system_matrix.reinit (sparsity_pattern);
        
        solution.reinit (dof_handler.n_dofs());
        system_rhs.reinit (dof_handler.n_dofs());
    }
    
    
    
    template <int dim>
    void HelmholtzProblem<dim>::assemble_system ()
    {
        QGauss<dim>   quadrature_formula(3);
        
        const unsigned int n_q_points    = quadrature_formula.size();
        
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        
        FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       cell_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        FEValues<dim>  fe_values (*fe, quadrature_formula,
                                  update_values   | update_gradients |
                                  update_quadrature_points | update_JxW_values);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            cell_matrix = 0;
            
            fe_values.reinit (cell);
            
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        cell_matrix(i,j) += ((fe_values.shape_grad(i,q_point) *
                                              fe_values.shape_grad(j,q_point)
                                              +
                                              fe_values.shape_value(i,q_point) *
                                              fe_values.shape_value(j,q_point)) *
                                             fe_values.JxW(q_point));
                }
            
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    system_matrix.add (local_dof_indices[i],
                                       local_dof_indices[j],
                                       cell_matrix(i,j));
                
            }
        }
        
        hanging_node_constraints.condense (system_matrix);
        
        std::map<types::global_dof_index,double> boundary_values;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  0,
                                                  Solution<dim>(),
                                                  boundary_values);
        MatrixTools::apply_boundary_values (boundary_values,
                                            system_matrix,
                                            solution,
                                            system_rhs);
    }
    
    
    
    template <int dim>
    void HelmholtzProblem<dim>::assemble_rhs ()
    {
        QGauss<dim>   quadrature_formula(3);
        QGauss<dim-1> face_quadrature_formula(3);
        
        const unsigned int n_q_points    = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();
        
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        
        Vector<double>       cell_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        FEValues<dim>  fe_values (*fe, quadrature_formula,
                                  update_values   | update_gradients |
                                  update_quadrature_points | update_JxW_values);
        
        FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
                                          update_values         | update_quadrature_points  |
                                          update_normal_vectors | update_JxW_values);
        
        const RightHandSide<dim> right_hand_side;
        std::vector<double>  rhs_values (n_q_points);
        
        const Solution<dim> exact_solution;
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            cell_rhs = 0;
            
            fe_values.reinit (cell);
            
            right_hand_side.value_list (fe_values.get_quadrature_points(),
                                        rhs_values);
            
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    cell_rhs(i) += (fe_values.shape_value(i,q_point) *
                                    rhs_values [q_point] *
                                    fe_values.JxW(q_point));
                }
            
            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                if (cell->face(face_number)->at_boundary()
                    &&
                    (cell->face(face_number)->boundary_id() == 1))
                {
                    fe_face_values.reinit (cell, face_number);
                    
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                        const double neumann_value
                        = (exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *
                           fe_face_values.normal_vector(q_point));
                        
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                            cell_rhs(i) += (neumann_value *
                                            fe_face_values.shape_value(i,q_point) *
                                            fe_face_values.JxW(q_point));
                    }
                }
            
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }
        }
        
        hanging_node_constraints.condense (system_rhs);
        
        std::map<types::global_dof_index,double> boundary_values;

    }
    

    
    template <int dim>
    void HelmholtzProblem<dim>::solve ()
    {
        SolverControl           solver_control (1000, 1e-12);
        SolverCG<>              cg (solver_control);
        
        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(system_matrix, 1.2);
        
        cg.solve (system_matrix, solution, system_rhs,
                  preconditioner);
        
        hanging_node_constraints.distribute (solution);
    }
    
    
    
    template <int dim>
    void HelmholtzProblem<dim>::refine_grid ()
    {
        switch (refinement_mode)
        {
            case global_refinement:
            {
                triangulation.refine_global (1);
                break;
            }
                
            case adaptive_refinement:
            {
                Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
                
                KellyErrorEstimator<dim>::estimate (dof_handler,
                                                    QGauss<dim-1>(3),
                                                    typename FunctionMap<dim>::type(),
                                                    solution,
                                                    estimated_error_per_cell);
                
                GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                                 estimated_error_per_cell,
                                                                 0.3, 0.03);
                
                triangulation.execute_coarsening_and_refinement ();
                
                break;
            }
                
            default:
            {
                Assert (false, ExcNotImplemented());
            }
        }
    }
    
    
    
    template <int dim>
    void HelmholtzProblem<dim>::process_solution (const unsigned int cycle)
    {
        Vector<float> difference_per_cell (triangulation.n_active_cells());
        VectorTools::integrate_difference (dof_handler,
                                           solution,
                                           Solution<dim>(),
                                           difference_per_cell,
                                           QGauss<dim>(3),
                                           VectorTools::L2_norm);
        const double L2_error = difference_per_cell.l2_norm();
        
        VectorTools::integrate_difference (dof_handler,
                                           solution,
                                           Solution<dim>(),
                                           difference_per_cell,
                                           QGauss<dim>(3),
                                           VectorTools::H1_seminorm);
        const double H1_error = difference_per_cell.l2_norm();
        
        const QTrapez<1>     q_trapez;
        const QIterated<dim> q_iterated (q_trapez, 5);
        VectorTools::integrate_difference (dof_handler,
                                           solution,
                                           Solution<dim>(),
                                           difference_per_cell,
                                           q_iterated,
                                           VectorTools::Linfty_norm);
        const double Linfty_error = difference_per_cell.linfty_norm();
        
        const unsigned int n_active_cells=triangulation.n_active_cells();
        const unsigned int n_dofs=dof_handler.n_dofs();
        
        std::cout << "Cycle " << cycle << ':'
        << std::endl
        << "   Number of active cells:       "
        << n_active_cells
        << std::endl
        << "   Number of degrees of freedom: "
        << n_dofs
        << std::endl;
        
        convergence_table.add_value("cycle", cycle);
        convergence_table.add_value("cells", n_active_cells);
        convergence_table.add_value("dofs", n_dofs);
        convergence_table.add_value("L2", L2_error);
        convergence_table.add_value("H1", H1_error);
        convergence_table.add_value("Linfty", Linfty_error);
        
        
    }
    
    template <int dim>
    void HelmholtzProblem<dim>::table()
    {
        
        std::string vtk_filename;
        switch (refinement_mode)
        {
            case global_refinement:
                vtk_filename = "solution-global";
                break;
            case adaptive_refinement:
                vtk_filename = "solution-adaptive";
                break;
            default:
                Assert (false, ExcNotImplemented());
        }
        
        switch (fe->degree)
        {
            case 1:
                vtk_filename += "-q1";
                break;
            case 2:
                vtk_filename += "-q2";
                break;
                
            default:
                Assert (false, ExcNotImplemented());
        }
        
        vtk_filename += ".vtk";
        std::ofstream output (vtk_filename.c_str());
        
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (solution, "solution");
        
        data_out.build_patches (fe->degree);
        data_out.write_vtk (output);
        
        
        
        convergence_table.set_precision("L2", 3);
        convergence_table.set_precision("H1", 3);
        convergence_table.set_precision("Linfty", 3);
        
        convergence_table.set_scientific("L2", true);
        convergence_table.set_scientific("H1", true);
        convergence_table.set_scientific("Linfty", true);
        
        convergence_table.set_tex_caption("cells", "\\# cells");
        convergence_table.set_tex_caption("dofs", "\\# dofs");
        convergence_table.set_tex_caption("L2", "$L^2$-error");
        convergence_table.set_tex_caption("H1", "$H^1$-error");
        convergence_table.set_tex_caption("Linfty", "$L^\\infty$-error");
        
        convergence_table.set_tex_format("cells", "r");
        convergence_table.set_tex_format("dofs", "r");
        
        std::cout << std::endl;
        convergence_table.write_text(std::cout);
        
        std::string error_filename = "error";
        switch (refinement_mode)
        {
            case global_refinement:
                error_filename += "-global";
                break;
            case adaptive_refinement:
                error_filename += "-adaptive";
                break;
            default:
                Assert (false, ExcNotImplemented());
        }
        
        switch (fe->degree)
        {
            case 1:
                error_filename += "-q1";
                break;
            case 2:
                error_filename += "-q2";
                break;
            default:
                Assert (false, ExcNotImplemented());
        }
        
        error_filename += ".tex";
        std::ofstream error_table_file(error_filename.c_str());
        
        convergence_table.write_tex(error_table_file);
        
        
        
        if (refinement_mode==global_refinement)
        {
            convergence_table.add_column_to_supercolumn("cycle", "n cells");
            convergence_table.add_column_to_supercolumn("cells", "n cells");
            
            std::vector<std::string> new_order;
            new_order.push_back("n cells");
            new_order.push_back("H1");
            new_order.push_back("L2");
            convergence_table.set_column_order (new_order);
            
            convergence_table
            .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
            convergence_table
            .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
            convergence_table
            .evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate);
            convergence_table
            .evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);
            
            std::cout << std::endl;
            convergence_table.write_text(std::cout);
            
            std::string conv_filename = "convergence";
            switch (refinement_mode)
            {
                case global_refinement:
                    conv_filename += "-global";
                    break;
                case adaptive_refinement:
                    conv_filename += "-adaptive";
                    break;
                default:
                    Assert (false, ExcNotImplemented());
            }
            switch (fe->degree)
            {
                case 1:
                    conv_filename += "-q1";
                    break;
                case 2:
                    conv_filename += "-q2";
                    break;
                default:
                    Assert (false, ExcNotImplemented());
            }
            conv_filename += ".tex";
            
            std::ofstream table_file(conv_filename.c_str());
            convergence_table.write_tex(table_file);
        }

    }
    
    
    
    template <int dim>
    void HelmholtzProblem<dim>::run ()
    {
        const unsigned int n_cycles = (refinement_mode==global_refinement)?5:9;
        
        for (unsigned int cycle=0; cycle<data::no_cycles; ++cycle)
        {
            if (cycle == 0)
            {
                
                GridGenerator::hyper_cube (triangulation, -1, 1);
                triangulation.refine_global (3);
                typename Triangulation<dim>::cell_iterator
                cell = triangulation.begin (),
                endc = triangulation.end();
                for (; cell!=endc; ++cell)
                    for (unsigned int face_number=0;
                         face_number<GeometryInfo<dim>::faces_per_cell;
                         ++face_number)
                        if ((std::fabs(cell->face(face_number)->center()(0) - (-1)) < 1e-12)
                            ||
                            (std::fabs(cell->face(face_number)->center()(1) - (-1)) < 1e-12))
                            cell->face(face_number)->set_boundary_id (1);
               /* {
                    std::vector<unsigned int> subdivisions (dim, 1);
                    subdivisions[0] = 4;
                    
                    const Point<dim> bottom_left = (dim == 2 ?
                                                    Point<dim>(0,data::bottom) :
                                                    Point<dim>(-2,0,-1));
                    const Point<dim> top_right   = (dim == 2 ?
                                                    Point<dim>(data::right,data::top) :
                                                    Point<dim>(0,1,0));
                    
                    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                                               subdivisions,
                                                               bottom_left,
                                                               top_right);
                }
                
                triangulation.refine_global (data::global_refinement_level);
                typename Triangulation<dim>::cell_iterator
                cell = triangulation.begin (),
                endc = triangulation.end();
                for (; cell!=endc; ++cell)
                    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                        if (cell->face(f)->center()[dim-1] == data::top)
                            cell->face(f)->set_all_boundary_ids(1);
                        else if (cell->face(f)->center()[dim-1] == data::bottom)
                            cell->face(f)->set_all_boundary_ids(2);*/
            }
            else
                refine_grid ();
            
            
            setup_system ();
            
            assemble_system ();
            assemble_rhs ();
            
            std::map<types::global_dof_index,double> boundary_values;
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      0,
                                                      Solution<dim>(),
                                                      boundary_values);
            MatrixTools::apply_boundary_values (boundary_values,
                                                system_matrix,
                                                solution,
                                                system_rhs);
            solve ();
            
            process_solution (cycle);
        }
        
        table();
        
    }
    
}


int main ()
{
    const unsigned int dim = 2;
    
    try
    {
        using namespace dealii;
        using namespace Step7;
        
        /*{
            std::cout << "Solving with Q1 elements, adaptive refinement" << std::endl
            << "=============================================" << std::endl
            << std::endl;
            
            FE_Q<dim> fe(1);
            HelmholtzProblem<dim>
            helmholtz_problem_2d (fe, HelmholtzProblem<dim>::adaptive_refinement);
            
            helmholtz_problem_2d.run ();
            
            std::cout << std::endl;
        }*/
        
        {
            std::cout << "Solving with Q1 elements, global refinement" << std::endl
            << "===========================================" << std::endl
            << std::endl;
            
            FE_Q<dim> fe(1);
            HelmholtzProblem<dim>
            helmholtz_problem_2d (fe, HelmholtzProblem<dim>::global_refinement);
            
            helmholtz_problem_2d.run ();
            
            std::cout << std::endl;
        }
        /*
        {
            std::cout << "Solving with Q2 elements, global refinement" << std::endl
            << "===========================================" << std::endl
            << std::endl;
            
            FE_Q<dim> fe(2);
            HelmholtzProblem<dim>
            helmholtz_problem_2d (fe, HelmholtzProblem<dim>::global_refinement);
            
            helmholtz_problem_2d.run ();
            
            std::cout << std::endl;
        }
        {
            std::cout << "Solving with Q2 elements, adaptive refinement" << std::endl
            << "===========================================" << std::endl
            << std::endl;
            
            FE_Q<dim> fe(2);
            HelmholtzProblem<dim>
            helmholtz_problem_2d (fe, HelmholtzProblem<dim>::adaptive_refinement);
            
            helmholtz_problem_2d.run ();
            
            std::cout << std::endl;
        }*/
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


