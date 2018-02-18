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
namespace Step22
{
    using namespace dealii;
    using namespace numbers;
    
    namespace data
    {
        const double eta = 1.0;
        const double lambda = 1.0;
        const double perm_const = 1.0;
        const double rho_f = 1.0;
        const double rho_r = 1.0;
        const double gamma = 1.0;
        const double pr_constant = 0.0;
        
        const double top = 1.0;
        const double bottom = 0.0;
        const double left = 0.0;
        const double right = PI;
        
        const int dimension = 2;
        const int problem_degree = 1;
        const int refinement_level = 4;
        
        const double timestep = 1e-2;
        const double final_time = 5*timestep;
        const double error_time = 13;
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
        void output_results () const;
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
        
    };
    
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
        values(0) = 0;
        values(1) = -(p[1]*p[1]*p[1] + 4*p[1]);
        values(2) = 0;
    }
    
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
        values(0) = 0;
        values(1) = -(p[1]*p[1]*p[1] + 4*p[1]);
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
        values(1) = 4.0 + 3*p[1]*p[1];
        values(2) = 2*p[1];
    }
    

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
        {
            constraints.clear ();
            
            FEValuesExtractors::Vector velocities(0);
            FEValuesExtractors::Scalar pressure (dim);
            
            DoFTools::make_hanging_node_constraints (dof_handler,
                                                     constraints);
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      1,
                                                      RockExactSolution<dim>(),
                                                      constraints,
                                                      fe.component_mask(velocities));
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      2,
                                                      RockExactSolution<dim>(),
                                                      constraints,
                                                      fe.component_mask(velocities));
            
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
                                               update_values | update_normal_vectors |
                                               update_quadrature_points |update_JxW_values   );
        
        const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        const ExtraRHSRock<dim>          right_hand_side;
        std::vector<Vector<double> >      rhs_values (n_q_points,
                                                      Vector<double>(dim+1));
        
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
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            local_matrix = 0;
            local_rhs = 0;
            right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                              rhs_values);
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                    symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                    div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                    phi_p[k]         = fe_values[pressure].value (k, q);
                }
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j=0; j<=i; ++j)
                    {
                        local_matrix(i,j) += (2 * (symgrad_phi_u[i] * symgrad_phi_u[j])
                                              - div_phi_u[i] * phi_p[j]
                                              - phi_p[i] * div_phi_u[j])
                        * fe_values.JxW(q);
                    }
                    const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                    
                    local_rhs(i) += fe_values.shape_value(i,q) *
                                        rhs_values[q](component_i) *
                                                        fe_values.JxW(q);
                }
            }
            
//            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
//                if (cell->face(face_number)->at_boundary()
//                    &&
//                    (cell->face(face_number)->boundary_id() == 1))
//                {
//                    fe_face_values.reinit (cell, face_number);
//
//                    topstress.vector_value_list(fe_face_values.get_quadrature_points(),
//                                                topstress_values);
//
//                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
//                    {
//
//                        for (unsigned int i=0; i<dofs_per_cell; ++i)
//                        {
//
//                            const unsigned int component_i = fe.system_to_component_index(i).first;
//
//                            local_rhs(i) += (
//                                             topstress_values[q_point](component_i)*
//                                             fe_face_values.
//                                             shape_value(i,q_point) *
//                                             fe_face_values.JxW(q_point));
//                        }
//                    }
//                }
//
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
//                            const unsigned int component_i = fe.system_to_component_index(i).first;
//                            local_rhs(i) += (bottomstress_values[q_point](component_i)*
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
            
            constraints.distribute_local_to_global (local_matrix, local_rhs,
                                                    local_dof_indices,
                                                    system_matrix, system_rhs);
        }
        // Need a condition as Dirichlet conditions as is makes the pressure only determined to a constant
        
        std::map<types::global_dof_index, double> pr_determination;
        {
            types::global_dof_index n_dofs = dof_handler.n_dofs();
            std::vector<bool> componentVector(dim + 1, false);
            componentVector[dim] = true;
            
            std::vector<bool> selected_dofs(n_dofs);
            std::set< types::boundary_id > boundary_ids;
            boundary_ids.insert(2);
            
            DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(componentVector),
                                            selected_dofs, boundary_ids);
            
            for (types::global_dof_index i = 0; i < n_dofs; i++)
            {
                if (selected_dofs[i]) pr_determination[i] = data::pr_constant;
            }
        }
        MatrixTools::apply_boundary_values(pr_determination,
                                           system_matrix, solution, system_rhs);
    }
    
    template <int dim>
    void StokesProblem<dim>::solve ()
    {
        SparseDirectUMFPACK  A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult (solution, system_rhs);
        constraints.distribute (solution);
    }
    
    template <int dim>
    void
    StokesProblem<dim>::output_results ()  const
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
        filename << "solution"
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
        std::vector<unsigned int> subdivisions (dim, 1);
        subdivisions[0] = 4;
        
        const Point<dim> bottom_left = (dim == 2 ?
                                        Point<dim>(data::left,data::bottom) :
                                        Point<dim>(-2,0,-1));
        const Point<dim> top_right   = (dim == 2 ?
                                        Point<dim>(data::right,data::top) :
                                        Point<dim>(0,1,0));
        
        GridGenerator::subdivided_hyper_rectangle (triangulation,
                                                   subdivisions, bottom_left, top_right);
        
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->center()[dim-1] == data::top)
                    cell->face(f)->set_all_boundary_ids(1);
                else if (cell->face(f)->center()[dim-1] == data::bottom)
                    cell->face(f)->set_all_boundary_ids(2);
        
           triangulation.refine_global (data::refinement_level);

        
   
            setup_dofs ();
            std::cout << "   Assembling..." << std::endl << std::flush;
            assemble_system ();
            std::cout << "   Solving..." << std::flush;
            solve ();
            output_results ();

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
