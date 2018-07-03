#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
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
#include <iostream>
#include <fstream>
#include <sstream>
namespace FullSolver
{
    using namespace dealii;
    using namespace numbers;
    
    namespace data
    {
        const double vr2_constant = 0.0;
        
        const double degree_vr = 2;
        const double degree_pr = 1;
        
        const double top = 1.0;
        const double bottom = 0.0;
        const double left = 0.0;
        const double right = 4.0;
        
        const int refinement_level = 12;
    }
    using namespace data;
    
    template <int dim>
    class FullMovingMesh
    {
    public:
        FullMovingMesh (const unsigned int degree_vr, const unsigned int degree_pr);
        void run ();
    private:
        void setup_dofs_rock ();
        void assemble_system_rock ();
        void solve_rock ();
        void output_results () const;
        void compute_errors ();
        const unsigned int   degree_vr;
        const unsigned int   degree_pr;
        Triangulation<dim>   triangulation;
        FESystem<dim>        fe_rock;
        DoFHandler<dim>      dof_handler_rock;
        ConstraintMatrix     constraints_rock;
        BlockSparsityPattern      sparsity_pattern_rock;
        BlockSparseMatrix<double> system_matrix_rock;
        BlockVector<double> solution_rock;
        BlockVector<double> system_rhs_rock;
        
    };
    
    // n.(pI-2e(u)) for the top
    template <int dim>
    class RockTopStress : public Function<dim>
    {
    public:
        RockTopStress () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p, Vector<double>   &value) const;
    };
    
    template <int dim>
    void
    RockTopStress<dim>::vector_value (const Point<dim> &p, Vector<double>   &values) const
    {
        const double phi = 0.5*p[1];
        values(0) = 0;
        values(1) = -(1-phi)*(p[1]*p[1]*p[1] + 4*p[1]);
        values(2) = 0;
    }
    
    // n.(pI-2e(u)) for the top
    template <int dim>
    class RockBottomStress : public Function<dim>
    {
    public:
        RockBottomStress () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p, Vector<double>   &value) const;
    };
    
    template <int dim>
    void
    RockBottomStress<dim>::vector_value (const Point<dim> &p, Vector<double>   &values) const
    {
        const double phi = 0.5*p[1];
        values(0) = 0;
        values(1) = (1-phi)*(p[1]*p[1]*p[1] + 4*p[1]);
        values(2) = 0;
    }
    
    template <int dim>
    class RockExactSolution : public Function<dim>
    {
    public:
        RockExactSolution () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    
    template <int dim>
    void
    RockExactSolution<dim>::vector_value (const Point<dim> &p,
                                          Vector<double>   &values) const
    {
        values(0) = 0;
        values(1) = -p[1]*p[1];
        values(2) = p[1]*p[1]*p[1];
    }
    
    // Extra terms you get on the right hand side from manufactured solutions
    template <int dim>
    class ExtraRHSRock : public Function<dim>
    {
    public:
        ExtraRHSRock () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p, Vector<double>   &value) const;
    };
    
    template <int dim>
    void
    ExtraRHSRock<dim>::vector_value (const Point<dim> &p, Vector<double>   &values) const
    {
        
        values(0) = 0;
        values(1) = -4+4*p[1]-3*p[1]*p[1]+2*p[1]*p[1]*p[1];
        values(2) = 3/2*p[1]*p[1]-2*p[1];
    }
    template <int dim>
    class ExactSolution_phi : public Function<dim>
    {
    public:
        ExactSolution_phi () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double ExactSolution_phi<dim>::value (const Point<dim>  &p,
                                          const unsigned int /*component*/) const
    {
        return  0.5*p[1];
        
    }
    
    template <int dim>
    class BoundaryValues : public Function<dim>
    {
    public:
        BoundaryValues () : Function<dim>(dim+1) {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    template <int dim>
    double
    BoundaryValues<dim>::value (const Point<dim>  &p,
                                const unsigned int component) const
    {
        Assert (component < this->n_components,
                ExcIndexRange (component, 0, this->n_components));
        return 0;
    }
    template <int dim>
    void
    BoundaryValues<dim>::vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const
    {
        for (unsigned int c=0; c<this->n_components; ++c)
            values(c) = BoundaryValues<dim>::value (p, c);
    }

    
    
    template <int dim>
    FullMovingMesh<dim>::FullMovingMesh (const unsigned int degree_vr, const unsigned int degree_pr)
    :
    degree_vr (degree_vr),
    degree_pr (degree_pr),
    triangulation (Triangulation<dim>::maximum_smoothing),
    fe_rock (FE_Q<dim>(degree_vr), dim,
             FE_Q<dim>(degree_pr), 1),
    dof_handler_rock (triangulation)
    {}
    template <int dim>
    void FullMovingMesh<dim>::setup_dofs_rock ()
    {
        
        system_matrix_rock.clear ();
        dof_handler_rock.distribute_dofs (fe_rock);
        DoFRenumbering::Cuthill_McKee (dof_handler_rock);
        std::vector<unsigned int> block_component (dim+1,0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise (dof_handler_rock, block_component);
        {
            constraints_rock.clear ();
            
            FEValuesExtractors::Vector velocities(0);
            FEValuesExtractors::Scalar pressure (dim);
            
            DoFTools::make_hanging_node_constraints (dof_handler_rock,
                                                     constraints_rock);

            // These are the conditions for the side boundary ids 0 (no flux)
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (0);
            VectorTools::compute_no_normal_flux_constraints (dof_handler_rock, 0,
                                                             no_normal_flux_boundaries,
                                                             constraints_rock);
            
                VectorTools::interpolate_boundary_values (dof_handler_rock,
                                                          2,
                                                          BoundaryValues<dim>(),
                                                          constraints_rock,
                                                          fe_rock.component_mask(velocities));
        }
        constraints_rock.close ();
        
        std::vector<types::global_dof_index> dofs_per_block (2);
        DoFTools::count_dofs_per_block (dof_handler_rock, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0],
        n_p = dofs_per_block[1];
        std::cout << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Number of degrees of freedom: "
        << dof_handler_rock.n_dofs()
        << " (" << n_u << '+' << n_p << ')'
        << std::endl;
        
        {
            BlockDynamicSparsityPattern dsp (2,2);
            dsp.block(0,0).reinit (n_u, n_u);
            dsp.block(1,0).reinit (n_p, n_u);
            dsp.block(0,1).reinit (n_u, n_p);
            dsp.block(1,1).reinit (n_p, n_p);
            dsp.collect_sizes();
            DoFTools::make_sparsity_pattern (dof_handler_rock, dsp, constraints_rock, false);
            sparsity_pattern_rock.copy_from (dsp);
        }
        
        system_matrix_rock.reinit (sparsity_pattern_rock);
        solution_rock.reinit (2);
        solution_rock.block(0).reinit (n_u);
        solution_rock.block(1).reinit (n_p);
        solution_rock.collect_sizes ();
        system_rhs_rock.reinit (2);
        system_rhs_rock.block(0).reinit (n_u);
        system_rhs_rock.block(1).reinit (n_p);
        system_rhs_rock.collect_sizes ();
    }
    
    template <int dim>
    void FullMovingMesh<dim>::assemble_system_rock ()
    {
        system_matrix_rock=0;
        system_rhs_rock=0;
        QGauss<dim>   quadrature_formula(degree_vr+2);
        QGauss<dim-1> face_quadrature_formula(degree_vr+2);
        
        FEValues<dim> fe_values_rock (fe_rock, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);
        FEFaceValues<dim> fe_face_values_rock ( fe_rock, face_quadrature_formula,
                                               update_values | update_normal_vectors |
                                               update_quadrature_points |update_JxW_values   );
        
        const unsigned int   dofs_per_cell   = fe_rock.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        const ExtraRHSRock<dim>          right_hand_side;
        std::vector<Vector<double> >      rhs_values (n_q_points,
                                                      Vector<double>(dim+1));
        const ExactSolution_phi<dim>    phi_solution;
        std::vector<double>             phi_values(n_q_points);
        
        const RockTopStress<dim>        topstress;
        const RockBottomStress<dim>     bottomstress;
        std::vector<Vector<double> >      topstress_values (n_face_q_points, Vector<double>      (dim+1));
        std::vector<Vector<double> >      bottomstress_values (n_face_q_points, Vector<double>      (dim+1));
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);
        std::vector<double>                  div_phi_u   (dofs_per_cell);
        std::vector<double>                  phi_p       (dofs_per_cell);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_rock.begin_active(),
        endc = dof_handler_rock.end();
        for (; cell!=endc; ++cell)
        {
            fe_values_rock.reinit (cell);
            local_matrix = 0;
            local_rhs = 0;
            right_hand_side.vector_value_list(fe_values_rock.get_quadrature_points(),
                                              rhs_values);
            phi_solution.value_list(fe_values_rock.get_quadrature_points(), phi_values);
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                    symgrad_phi_u[k] = fe_values_rock[velocities].symmetric_gradient (k, q);
                    div_phi_u[k]     = fe_values_rock[velocities].divergence (k, q);
                    phi_p[k]         = fe_values_rock[pressure].value (k, q);
                }
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j=0; j<=i; ++j)
                    { // matrix assembly
                        local_matrix(i,j) += (-2 * (1-phi_values[q])*(symgrad_phi_u[i] * symgrad_phi_u[j])
                                              + (1-phi_values[q])*div_phi_u[i] * phi_p[j]
                                              + (1-phi_values[q])*phi_p[i] * div_phi_u[j])
                        * fe_values_rock.JxW(q);
                    }
                    
                    const unsigned int component_i = fe_rock.system_to_component_index(i).first;
                    
                    // Add the extra terms on RHS
                    local_rhs(i) +=
                    fe_values_rock.shape_value(i,q) *
                    rhs_values[q](component_i) *
                    fe_values_rock.JxW(q);
                }
            }
            
            //            //Neumann Stress conditions on top boundary
            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                if (cell->face(face_number)->at_boundary()
                    &&
                    (cell->face(face_number)->boundary_id() == 1))
                {
                    fe_face_values_rock.reinit (cell, face_number);
                    
                    topstress.vector_value_list(fe_face_values_rock.get_quadrature_points(),
                                                topstress_values);
                    
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                        
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                            
                            const unsigned int component_i = fe_rock.system_to_component_index(i).first;
                            
                            local_rhs(i) += (-topstress_values[q_point](component_i)*
                                             fe_face_values_rock.
                                             shape_value(i,q_point) *
                                             fe_face_values_rock.JxW(q_point));
                        }
                    }
                }
            
            // Neumann Stress conditions on bottom boundary
            //            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
            //                if (cell->face(face_number)->at_boundary()
            //                    &&
            //                    (cell->face(face_number)->boundary_id() == 2))
            //                {
            //
            //                    fe_face_values.reinit (cell, face_number);
            //
            //                    bottomstress.vector_value_list(fe_face_values.get_quadrature_points(),
            //                                                   bottomstress_values);
            //
            //                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
            //                    {
            //                        for (unsigned int i=0; i<dofs_per_cell; ++i)
            //                        {
            //                            const unsigned int component_i = fe_rock.system_to_component_index(i).first;
            //                            local_rhs(i) += (-bottomstress_values[q_point](component_i)*
            //                                             fe_face_values.
            //                                             shape_value(i,q_point) *
            //                                             fe_face_values.JxW(q_point));
            //                        }
            //                    }
            //                }
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                    local_matrix(i,j) = local_matrix(j,i);
            cell->get_dof_indices (local_dof_indices);
            
            constraints_rock.distribute_local_to_global (local_matrix, local_rhs,
                                                         local_dof_indices,
                                                         system_matrix_rock, system_rhs_rock);
        }
        //        std::map<types::global_dof_index, double> vr_determination;
        //        {
        //            types::global_dof_index n_dofs = dof_handler_rock.n_dofs();
        //            std::vector<bool> componentVector(dim + 1, false);
        //            componentVector[dim] = true;
        //
        //            std::vector<bool> selected_dofs(n_dofs);
        //            std::set< types::boundary_id > boundary_ids;
        //            boundary_ids.insert(1);
        //
        //            DoFTools::extract_boundary_dofs(dof_handler_rock, ComponentMask(componentVector),
        //                                            selected_dofs, boundary_ids);
        //
        //            for (types::global_dof_index i = 0; i < n_dofs; i++)
        //            {
        //                if (selected_dofs[i]) vr_determination[i] = 1.0;
        //            }
        //        }
        //        MatrixTools::apply_boundary_values(vr_determination,
        //                                           system_matrix_rock, solution_rock, system_rhs_rock);
        //        {
        //            std::map<types::global_dof_index, double> vr_determination;
        //            {
        //                types::global_dof_index n_dofs = dof_handler_rock.n_dofs();
        //                std::vector<bool> componentVector(dim + 1, false);
        //                componentVector[dim] = true;
        //
        //                std::vector<bool> selected_dofs(n_dofs);
        //                std::set< types::boundary_id > boundary_ids;
        //                boundary_ids.insert(2);
        //
        //                DoFTools::extract_boundary_dofs(dof_handler_rock, ComponentMask(componentVector),
        //                                                selected_dofs, boundary_ids);
        //
        //                for (types::global_dof_index i = 0; i < n_dofs; i++)
        //                {
        //                    if (selected_dofs[i]) vr_determination[i] = vr2_constant;
        //                }
        //            }
        //            MatrixTools::apply_boundary_values(vr_determination,
        //                                               system_matrix_rock, solution_rock, system_rhs_rock);
        //        }
    }
    
    template <int dim>
    void FullMovingMesh<dim>::solve_rock ()
    {
        SparseDirectUMFPACK  A_direct;
        A_direct.initialize(system_matrix_rock);
        A_direct.vmult (solution_rock, system_rhs_rock);
        constraints_rock.distribute (solution_rock);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::compute_errors ()
    {
        {
            const ComponentSelectFunction<dim> pressure_mask (dim, dim+1);
            const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);
            RockExactSolution<dim> exact_solution;
            
            Vector<double> cellwise_errors (triangulation.n_active_cells());
            
            QTrapez<1>     q_trapez;
            QIterated<dim> quadrature (q_trapez, degree_pr+2);
            
            VectorTools::integrate_difference (dof_handler_rock, solution_rock, exact_solution,
                                               cellwise_errors, quadrature,
                                               VectorTools::L2_norm,
                                               &pressure_mask);
            const double p_l2_error = cellwise_errors.l2_norm();
            
            VectorTools::integrate_difference (dof_handler_rock, solution_rock, exact_solution,
                                               cellwise_errors, quadrature,
                                               VectorTools::L2_norm,
                                               &velocity_mask);
            const double u_l2_error = cellwise_errors.l2_norm();
            std::cout << "   Errors: ||e_pr||_L2  = " << p_l2_error
            << ",  " << std::endl << "           ||e_vr||_L2  = " << u_l2_error
            << std::endl;
        }
        
    }
    
    
    template <int dim>
    void
    FullMovingMesh<dim>::output_results ()  const
    {
        std::vector<std::string> solution_names (dim, "velocity");
        solution_names.push_back ("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation
        (dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation
        .push_back (DataComponentInterpretation::component_is_scalar);
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler_rock);
        data_out.add_data_vector (solution_rock, solution_names,
                                  DataOut<dim>::type_dof_data,
                                  data_component_interpretation);
        data_out.build_patches ();
        std::ostringstream filename;
        filename << "solution_rock"
        << ".vtk";
        std::ofstream output (filename.str().c_str());
        data_out.write_vtk (output);
    }
    
    
    template <int dim>
    void FullMovingMesh<dim>::run ()
    {
        std::vector<unsigned int> subdivisions (dim, 1);
        subdivisions[0] = 3;
        
        const Point<dim> bottom_left = (dim == 2 ?
                                        Point<dim>(data::left,data::bottom) :
                                        Point<dim>(-2,0,-1));
        const Point<dim> top_right   = (dim == 2 ?
                                        Point<dim>(data::right,data::top) :
                                        Point<dim>(0,1,0));
        
        GridGenerator::subdivided_hyper_rectangle (triangulation, subdivisions, bottom_left, top_right);
        
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->center()[dim-1] == data::top)
                    cell->face(f)->set_all_boundary_ids(1);
                else if (cell->face(f)->center()[dim-1] == data::bottom)
                    cell->face(f)->set_all_boundary_ids(2);
        
        triangulation.refine_global (data::refinement_level);
        
        
        
        setup_dofs_rock ();
        std::cout << "   Assembling..." << std::endl << std::flush;
        assemble_system_rock ();
        std::cout << "   Solving..." << std::endl << std::flush;
        solve_rock ();
        output_results ();
        compute_errors ();
        
    }
}
int main ()
{
    try
    {
        using namespace dealii;
        using namespace FullSolver;
        FullMovingMesh<2> flow_problem(degree_vr, degree_pr);
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


