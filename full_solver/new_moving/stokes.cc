#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
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
#include <iostream>
#include <fstream>
#include <sstream>
namespace Step22
{
    using namespace dealii;
    using namespace numbers;
    
    namespace data
    {
        const int refinement_level = 4;
        const double top = 1.0;
        const double bottom = 0.0;
        const double left = 0.0;
        const double right = PI;
        
        const double phi = 0.7;
        const double pr_constant = (1-phi)*bottom*bottom*bottom;
        
        const double timestep = 0.2;
        const double initial_time = 0.0;
        int timestep_number = 0;
        int total_timesteps = 1;
        double present_time = initial_time + timestep*total_timesteps;

    }
    using namespace data;
    
    template <int dim>
    class StokesProblem
    {
    public:
        StokesProblem (const unsigned int degree_rock);
        void run ();
    private:
        void setup_dofs ();
        void assemble_system_rock ();
        void solve_rock ();
        void output_results_rock () const;
        void move_mesh ();
        void print_mesh ();
        void compute_errors_rock ();

        const unsigned int        degree_rock;
        Triangulation<dim>        triangulation;
        FESystem<dim>             fe_rock;
        DoFHandler<dim>           dof_handler_rock;
        ConstraintMatrix          constraints_rock;
        BlockSparsityPattern      sparsity_pattern_rock;
        BlockSparseMatrix<double> system_matrix_rock;
        BlockVector<double>       solution_rock;
        BlockVector<double>       system_rhs_rock;
        Vector<double>            displacement;
        
    };
  
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
        values(1) = -(1-phi)*p[1]*p[1];
        values(2) = (1-phi)*p[1]*p[1]*p[1];
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
        values(1) = (1-phi)*(4.0 + 3*p[1]*p[1]);
        values(2) = (1-phi)*(2*p[1]);
    }
    

    template <int dim>
    StokesProblem<dim>::StokesProblem (const unsigned int degree_rock)
    :
    degree_rock (degree_rock),
    triangulation (Triangulation<dim>::maximum_smoothing),
    fe_rock (FE_Q<dim>(degree_rock+1), dim,
        FE_Q<dim>(degree_rock), 1),
    dof_handler_rock (triangulation)
    {}
    
    template <int dim>
    void StokesProblem<dim>::setup_dofs ()
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
            //            Testing if Dirichlet conditions work: use the following for top boundary 1 and bottom boundary id 2
            
            VectorTools::interpolate_boundary_values (dof_handler_rock,
                                                      1,
                                                      RockExactSolution<dim>(),
                                                      constraints_rock,
                                                      fe_rock.component_mask(velocities));
            VectorTools::interpolate_boundary_values (dof_handler_rock,
                                                      2,
                                                      RockExactSolution<dim>(),
                                                      constraints_rock,
                                                      fe_rock.component_mask(velocities));
            
            // These are the conditions for the side boundary ids 0 (no flux)
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (0);
            VectorTools::compute_no_normal_flux_constraints (dof_handler_rock, 0,
                                                             no_normal_flux_boundaries,
                                                             constraints_rock);
            
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
            BlockDynamicSparsityPattern dsp_rock (2,2);
            dsp_rock.block(0,0).reinit (n_u, n_u);
            dsp_rock.block(1,0).reinit (n_p, n_u);
            dsp_rock.block(0,1).reinit (n_u, n_p);
            dsp_rock.block(1,1).reinit (n_p, n_p);
            dsp_rock.collect_sizes();
            DoFTools::make_sparsity_pattern (dof_handler_rock, dsp_rock, constraints_rock, false);
            sparsity_pattern_rock.copy_from (dsp_rock);
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
        displacement.reinit(n_u);
    }
    
    template <int dim>
    void StokesProblem<dim>::assemble_system_rock ()
    {
        system_matrix_rock=0;
        system_rhs_rock=0;
        QGauss<dim>   quadrature_formula(degree_rock+2);
        
        FEValues<dim> fe_rock_values (fe_rock, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 update_gradients);
        
        const unsigned int   dofs_per_cell   = fe_rock.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();\
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        const ExtraRHSRock<dim>          right_hand_side;
        std::vector<Vector<double> >      rhs_values (n_q_points, Vector<double>(dim+1));
    
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
            fe_rock_values.reinit (cell);
            local_matrix = 0;
            local_rhs = 0;
            right_hand_side.vector_value_list(fe_rock_values.get_quadrature_points(),
                                              rhs_values);
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                    symgrad_phi_u[k] = fe_rock_values[velocities].symmetric_gradient (k, q);
                    div_phi_u[k]     = fe_rock_values[velocities].divergence (k, q);
                    phi_p[k]         = fe_rock_values[pressure].value (k, q);
                }
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j=0; j<=i; ++j)
                    { // matrix assembly
                        local_matrix(i,j) += (2 * (1-phi) *
                                              (symgrad_phi_u[i] * symgrad_phi_u[j])
                                              - (1-phi)*div_phi_u[i] * phi_p[j]
                                              - (1-phi)*phi_p[i] * div_phi_u[j])
                                                    * fe_rock_values.JxW(q);
                    }
                    
                    const unsigned int component_i =
                    fe_rock.system_to_component_index(i).first;
                    
                    // Add the extra terms on RHS
                    local_rhs(i) +=  fe_rock_values.shape_value(i,q) *
                    (1- phi)*rhs_values[q](component_i) *
                                                        fe_rock_values.JxW(q);
                }
            }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                    local_matrix(i,j) = local_matrix(j,i);
            
            cell->get_dof_indices (local_dof_indices);
            
            constraints_rock.distribute_local_to_global (local_matrix, local_rhs,
                                                    local_dof_indices,
                                                    system_matrix_rock, system_rhs_rock);
        }
        
        
        // Need a condition if using Dirichlet conditions as it makes the pressure only determined to a constant. uncomment when testing Dirichlet
//
        std::map<types::global_dof_index, double> pr_determination;
        {
            types::global_dof_index n_dofs = dof_handler_rock.n_dofs();
            std::vector<bool> componentVector(dim + 1, false);
            componentVector[dim] = true;

            std::vector<bool> selected_dofs(n_dofs);
            std::set< types::boundary_id > boundary_ids;
            boundary_ids.insert(2);

            DoFTools::extract_boundary_dofs(dof_handler_rock, ComponentMask(componentVector),
                                            selected_dofs, boundary_ids);

            for (types::global_dof_index i = 0; i < n_dofs; i++)
            {
                if (selected_dofs[i]) pr_determination[i] = pr_constant;
            }
        }
        MatrixTools::apply_boundary_values(pr_determination,
                                           system_matrix_rock, solution_rock, system_rhs_rock);
    }
    
    template <int dim>
    void StokesProblem<dim>::solve_rock ()
    {
        SparseDirectUMFPACK  A_direct;
        A_direct.initialize(system_matrix_rock);
        A_direct.vmult (solution_rock, system_rhs_rock);
        constraints_rock.distribute (solution_rock);
    }
    
    template <int dim>
    void
    StokesProblem<dim>::output_results_rock ()  const
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
    void StokesProblem<dim>::move_mesh ()
    {
        std::cout << "    Moving mesh..." << std::endl;
        
        std::vector<bool> vertex_touched (triangulation.n_vertices(),
                                          false);
        
        for (typename DoFHandler<dim>::active_cell_iterator
             cell = dof_handler_rock.begin_active ();
             cell != dof_handler_rock.end(); ++cell)
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                if (vertex_touched[cell->vertex_index(v)] == false)
                {
                    vertex_touched[cell->vertex_index(v)] = true;
                    Point<dim> vertex_displacement;
                    for (unsigned int d=0; d<dim; ++d)
                    {vertex_displacement[d]
                        = solution_rock.block(0)(cell->vertex_dof_index(v,d));
                    }
                    cell->vertex(v) += vertex_displacement*timestep;
                }
    }
    
    template <int dim>
    void StokesProblem<dim>::print_mesh ()
    {
            std::ofstream out ("grid-"
                               + Utilities::int_to_string(timestep_number, 3) +
                               ".eps");
            GridOut grid_out;
            grid_out.write_eps (triangulation, out);
    }
    
    template <int dim>
    void StokesProblem<dim>::compute_errors_rock ()
    {
        {
            const ComponentSelectFunction<dim> pressure_mask (dim, dim+1);
            const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);
            RockExactSolution<dim> exact_solution_rock;
            
            Vector<double> cellwise_errors (triangulation.n_active_cells());
            
            QTrapez<1>     q_trapez;
            QIterated<dim> quadrature (q_trapez, degree_rock+2);
            
            VectorTools::integrate_difference (dof_handler_rock, solution_rock, exact_solution_rock,
                                               cellwise_errors, quadrature,
                                               VectorTools::L2_norm,
                                               &pressure_mask);
            const double p_l2_error = cellwise_errors.l2_norm();
            
            VectorTools::integrate_difference (dof_handler_rock, solution_rock, exact_solution_rock,
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
    void StokesProblem<dim>::run ()
    {
        std::vector<unsigned int> subdivisions (dim, 1);
        subdivisions[0] = 4;
        
        const Point<dim> bottom_left = (dim == 2 ?
                                        Point<dim>( left, bottom) :
                                        Point<dim>(-2,0,-1));
        const Point<dim> top_right   = (dim == 2 ?
                                        Point<dim>( right, top) :
                                        Point<dim>(0,1,0));
        
        GridGenerator::subdivided_hyper_rectangle (triangulation,
                                                   subdivisions, bottom_left, top_right);
        
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->center()[dim-1] ==  top)
                    cell->face(f)->set_all_boundary_ids(1);
                else if (cell->face(f)->center()[dim-1] ==  bottom)
                    cell->face(f)->set_all_boundary_ids(2);
        
           triangulation.refine_global ( refinement_level);
       
        setup_dofs ();

        while (timestep_number < total_timesteps)
        {
            std::cout << "   Assembling at timestep number "
                            << timestep_number << "..." <<  std::endl << std::flush;
            assemble_system_rock ();
            std::cout << "   Solving at timestep number "
                        << timestep_number << "..." <<  std::endl << std::flush;
            solve_rock ();
            output_results_rock ();
            compute_errors_rock ();
            
            print_mesh ();
            move_mesh ();
            
            ++timestep_number;
        }
    }
}

int main ()
{
    try
    {
        using namespace dealii;
        using namespace Step22;
        StokesProblem<2> flow_problem(1);
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
