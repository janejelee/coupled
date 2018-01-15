/* ---------------------------------------------------------------------
Originally from veri_stokes.

Written to add the phis into the left hand side only.
Copied from stokes folder to implement the Neumann boundary conditions for the normal
component of the normal stress.

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
    using namespace numbers;
    
    namespace data
    {
        const double eta = 1.0;
        const double lambda = 1.0;
        const double perm_const = 1.0;
        const double rho_f = 1.0;
        const double rho_r = 1.0;
        const double gamma = 1.0;
        const double pr_constant = 2.5;
        
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
    class DAE
    {
    public:
        DAE (const unsigned int degree);
        ~DAE ();
        void run ();
        
    private:
        void make_grid();
        void setup_rock_dofs ();
        void setup_phi_dofs ();
        void assemble_rock_system ();
        void solve_rock_system ();
        void output_results ();
        void compute_errors ();
        void refine_mesh ();
        
        const unsigned int   pr_degree;
        
        Triangulation<dim>   triangulation;
        FESystem<dim>        rock_fe;
        DoFHandler<dim>      rock_dof_handler;
        
        ConstraintMatrix     rock_constraints;
        
        BlockSparsityPattern     	rock_sparsity_pattern;
        BlockSparseMatrix<double> 	rock_system_matrix;
        BlockVector<double> 		rock_solution;
        BlockVector<double> 		rock_system_rhs;
        
        void assemble_phi_system ();
        DoFHandler<dim>      phi_dof_handler;
        FE_Q<dim>            phi_fe;
        ConstraintMatrix     phi_hanging_node_constraints;
        SparsityPattern      phi_sparsity_pattern;
        SparseMatrix<double> phi_system_matrix;
        SparseMatrix<double> phi_mass_matrix;
        SparseMatrix<double> phi_nontime_matrix;
        Vector<double>       phi_solution;
        Vector<double>		 old_phi_solution;
        Vector<double>       phi_system_rhs;
        Vector<double>		 phi_nontime_rhs;
        
        double               time;
        double               time_step;
        unsigned int         timestep_number;

    };
    
    template <int dim>
    class RockTopStress : public Function<dim>
    {
    public:
        RockTopStress () : Function<dim>(1) {}
        
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double RockTopStress<dim>::value (const Point<dim>  &p,
                                      const unsigned int /*component*/) const
    {
        return p[1]*p[1]*p[1]+4*p[1];
    }
    
    template <int dim>
    class RockBottomStress : public Function<dim>
    {
    public:
        RockBottomStress () : Function<dim>(1) {}
        
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double RockBottomStress<dim>::value (const Point<dim>  &p,
                                         const unsigned int /*component*/) const
    {
        return p[1]*p[1]*p[1]+4*p[1];
    }
    
    template <int dim>
    class RockInitialRHS : public Function<dim>
    {
    public:
        RockInitialRHS () : Function<dim>(dim+1) {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    
    template <int dim>
    double
    RockInitialRHS<dim>::value (const Point<dim>  &p,
                                const unsigned int component) const
    {
        // LOG2
        if (component == 0)
            return 0;
        else if (component == 1)
            return -4-3*p[1]*p[1] + 0.0; //for initial time =0
        else if (component == dim)
            return -2*p[1] + 0.0; //for initial time =0
        return 0.0;
    }
    
    
    template <int dim>
    void
    RockInitialRHS<dim>::vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const
    {
        for (unsigned int c=0; c<this->n_components; ++c)
            values(c) = RockInitialRHS<dim>::value (p, c);
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
        const double time = this->get_time();
        
        values(0) = 0;
        values(1) = time*data::gamma*exp(-p[1])*(5*p[1]-5-data::rho_r+data::rho_f)
        -4 -3*p[1]*p[1]-data::rho_r;
        values(2) = -2*p[1]+ time*data::gamma*exp(-p[1]);
    }
    
    
    template <int dim>
    class PhiInitialFunction : public Function<dim>
    {
    public:
        PhiInitialFunction () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double PhiInitialFunction<dim>::value (const Point<dim>  &p,
                                           const unsigned int /*component*/) const
    {
        
        return 0.7 + p[0]*0.0;
    }
    
   
    template <int dim>
    DAE<dim>::DAE (const unsigned int degree)
    :
    pr_degree (degree),
    rock_fe (FE_Q<dim>(pr_degree+1), dim,
             FE_Q<dim>(pr_degree), 1),
    rock_dof_handler (triangulation),
    
    phi_dof_handler (triangulation),
    phi_fe (data::problem_degree)
    {}
    
    template <int dim>
    DAE<dim>::~DAE ()
    {
        phi_dof_handler.clear ();
    }
    
    
    template <int dim>
    void DAE<dim>::make_grid ()
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
    }
    
    template <int dim>
    void DAE<dim>::setup_phi_dofs()
    {
        phi_dof_handler.distribute_dofs (phi_fe);
        phi_hanging_node_constraints.clear ();
        DoFTools::make_hanging_node_constraints (phi_dof_handler,
                                                 phi_hanging_node_constraints);
        phi_hanging_node_constraints.close ();
        
        DynamicSparsityPattern dsp(phi_dof_handler.n_dofs(), phi_dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(phi_dof_handler,
                                        dsp, phi_hanging_node_constraints, /*keep_constrained_dofs = */ true);
        phi_sparsity_pattern.copy_from (dsp);
        
        phi_system_matrix.reinit (phi_sparsity_pattern);
        phi_mass_matrix.reinit (phi_sparsity_pattern);
        phi_nontime_matrix.reinit (phi_sparsity_pattern);
        phi_solution.reinit (phi_dof_handler.n_dofs());
        old_phi_solution.reinit(phi_dof_handler.n_dofs());
        phi_system_rhs.reinit (phi_dof_handler.n_dofs());
        phi_nontime_rhs.reinit (phi_dof_handler.n_dofs());
    }

    
    template <int dim>
    void DAE<dim>::setup_rock_dofs ()
    {
        rock_system_matrix.clear ();
        
        rock_dof_handler.distribute_dofs (rock_fe);
        DoFRenumbering::component_wise (rock_dof_handler);
        
        std::vector<unsigned int> block_component (dim+1,0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise (rock_dof_handler, block_component);
        
        rock_constraints.clear ();
        
        FEValuesExtractors::Vector velocities(0);
        FEValuesExtractors::Scalar pressure (dim);
        
        DoFTools::make_hanging_node_constraints (rock_dof_handler,
                                                 rock_constraints);
        
        std::set<types::boundary_id> no_normal_flux_boundaries;
        no_normal_flux_boundaries.insert (0);
        VectorTools::compute_no_normal_flux_constraints (rock_dof_handler, 0,
                                                         no_normal_flux_boundaries,
                                                         rock_constraints);
        
        rock_constraints.close ();
        
        std::vector<types::global_dof_index> dofs_per_block (2);
        DoFTools::count_dofs_per_block (rock_dof_handler, dofs_per_block, block_component);
        const unsigned int n_vr = dofs_per_block[0],
        n_pr = dofs_per_block[1];
        
        std::cout << "	Problem Degree: "
        << data::problem_degree
        << std::endl
        << "	Refinement level: "
        << data::refinement_level
        << std::endl
        << "	Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "	Total number of cells: "
        << triangulation.n_cells()
        << std::endl
        << "	Number of degrees of freedom in rock problem: "
        << rock_dof_handler.n_dofs()
        << " (" << n_vr << '+' << n_pr << ')'
        << std::endl;
        
        BlockDynamicSparsityPattern rock_dsp (2,2);
        
        rock_dsp.block(0,0).reinit (n_vr, n_vr);
        rock_dsp.block(1,0).reinit (n_pr, n_vr);
        rock_dsp.block(0,1).reinit (n_vr, n_pr);
        rock_dsp.block(1,1).reinit (n_pr, n_pr);
        
        rock_dsp.collect_sizes();
        
        DoFTools::make_sparsity_pattern (rock_dof_handler, rock_dsp, rock_constraints, false);
        rock_sparsity_pattern.copy_from (rock_dsp);
        
        rock_system_matrix.reinit (rock_sparsity_pattern);
        
        rock_solution.reinit (2);
        rock_solution.block(0).reinit (n_vr);
        rock_solution.block(1).reinit (n_pr);
        rock_solution.collect_sizes ();
        
        rock_system_rhs.reinit (2);
        rock_system_rhs.block(0).reinit (n_vr);
        rock_system_rhs.block(1).reinit (n_pr);
        rock_system_rhs.collect_sizes ();
    }
    
    template <int dim>
    void DAE<dim>::assemble_rock_system ()
    {
        rock_system_matrix=0;
        rock_system_rhs=0;
        
        QGauss<dim>   quadrature_formula(pr_degree+2);
        QGauss<dim-1> face_quadrature_formula(pr_degree+2);
        
        FEValues<dim> rock_fe_values (rock_fe, quadrature_formula,
                                      update_values    |  update_quadrature_points  |
                                      update_JxW_values | update_gradients);
        
        FEFaceValues<dim> rock_fe_face_values ( rock_fe, face_quadrature_formula,
                                               update_values | update_normal_vectors |
                                               update_quadrature_points |update_JxW_values   );
        
        FEValues<dim> phi_fe_values (phi_fe, quadrature_formula,
                                     update_values    | update_quadrature_points  |
                                     update_JxW_values | update_gradients);
        
//        FEValues<dim> pf_fe_values (pf_fe, quadrature_formula,
//                                    update_values    | update_quadrature_points  |
//                                    update_JxW_values | update_gradients);
//        
//        FEValues<dim> vf_fe_values (vf_fe, quadrature_formula,
//                                    update_values    | update_quadrature_points  |
//                                    update_JxW_values | update_gradients);
        
        const unsigned int   dofs_per_cell   = rock_fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        std::vector<double> boundary_values (n_face_q_points);
        const RockInitialRHS<dim>          initial_right_hand_side;
        std::vector<Vector<double> >      initial_rhs_values (n_q_points, Vector<double>(dim+1));
        const ExtraRHSRock<dim>      		  extraRHS;
        std::vector<Vector<double> >      extra_rhs_values (n_q_points, Vector<double>(dim+1));
        
        const RockTopStress<dim>        topstress;
        const RockBottomStress<dim>     bottomstress;
        std::vector<double>             topstress_values (n_face_q_points);
        std::vector<double>             bottomstress_values (n_face_q_points);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        std::vector<Tensor<1,dim>>    unitz_values (n_q_points);
//        std::vector<Tensor<1,dim>> 	  vf_values (n_q_points);
//        std::vector<double>			  pf_values (n_q_points);
        std::vector<double>			  phi_values (n_q_points);
//        std::vector<double>			  div_vf_values (n_q_points);
//        std::vector<Tensor<1,dim>>    grad_pf_values (n_q_points);
//        std::vector<Tensor<1,dim>>    grad_phi_values (n_q_points);
        
        std::vector<Tensor<1,dim>>           phi_u       (dofs_per_cell); // why is this a tensor?
        std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);
        std::vector<Tensor<2,dim> >          grad_phi_u (dofs_per_cell);
        std::vector<double>                  div_phi_u   (dofs_per_cell);
        std::vector<double>                  phi_p       (dofs_per_cell);
        
//        std::cout << "Assembling from beginning of system. Timstep number " <<
//        timestep_number << "." << std::endl;
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = rock_dof_handler.begin_active(),
        endc = rock_dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator
        phi_cell = phi_dof_handler.begin_active();
//        typename DoFHandler<dim>::active_cell_iterator
//        pf_cell = pf_dof_handler.begin_active();
//        typename DoFHandler<dim>::active_cell_iterator
//        vf_cell = vf_dof_handler.begin_active();
        for (; cell!=endc; ++cell, ++phi_cell/*, ++pf_cell, ++vf_cell*/)
        {
            rock_fe_values.reinit (cell);
            phi_fe_values.reinit (phi_cell);
//            pf_fe_values.reinit (pf_cell);
            local_matrix = 0;
            local_rhs = 0;
            
            initial_right_hand_side.vector_value_list(rock_fe_values.get_quadrature_points(),
                                                      initial_rhs_values);
            extraRHS.vector_value_list(rock_fe_values.get_quadrature_points(),
                                       extra_rhs_values);
            phi_fe_values.get_function_values (phi_solution, phi_values);
//            phi_fe_values.get_function_gradients (phi_solution, grad_phi_values);
//            pf_fe_values[pressure].get_function_values (pf_solution, pf_values);
//            pf_fe_values[pressure].get_function_gradients (pf_solution, grad_pf_values);
            
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                    phi_u[k]         = rock_fe_values[velocities].value (k, q);
                    grad_phi_u[k]    = rock_fe_values[velocities].gradient (k, q);
                    symgrad_phi_u[k] = rock_fe_values[velocities].symmetric_gradient(k,q);
                    div_phi_u[k]     = rock_fe_values[velocities].divergence (k, q);
                    phi_p[k]         = rock_fe_values[pressure].value (k, q);
                }
                
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                        local_matrix(i,j) += (-data::eta* (1.0-phi_values[q]) *2*
                                              (symgrad_phi_u[i]*symgrad_phi_u[j])
                                              + (1.0-phi_values[q])*div_phi_u[i] * phi_p[j]
                                              + (1.0-phi_values[q])*phi_p[i] * div_phi_u[j] )
                        * rock_fe_values.JxW(q);
                    }
                    
                    if (timestep_number == 0)
                    {
                        const unsigned int component_i = rock_fe.system_to_component_index(i).first;
                        local_rhs(i) += rock_fe_values.shape_value(i,q) *
                        (initial_rhs_values[q](component_i)
                         )*
                        rock_fe_values.JxW(q);
                    }
                    else
                    {
                        const unsigned int component_i = rock_fe.system_to_component_index(i).first;
                        
//                        local_rhs(i) += (grad_phi_values[q]*phi_u[i]
//                                         + ((1.0-phi_values[i])*data::rho_r + phi_values[q]*data::rho_f)
//                                         * unitz_values[q]*phi_u[i]
//                                         - phi_values[q]*phi_p[i]
//                                         + extra_rhs_values[q](component_i)*rock_fe_values.shape_value(i,q)
//                                         )*
//                        rock_fe_values.JxW(q);
                    }
                    
                    for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                        if (cell->face(face_number)->at_boundary()
                            &&
                            (cell->face(face_number)->boundary_id() == 1))
                        {
                            rock_fe_face_values.reinit (cell, face_number);
                            
                            const unsigned int component_i = rock_fe.system_to_component_index(i).first;
                            topstress.value_list (rock_fe_face_values.get_quadrature_points(),
                                                  topstress_values);
                            
                            for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                            {
                                
                                for (unsigned int i=0; i<dofs_per_cell; ++i)
                                {
                                    local_rhs(i) += (/*rock_fe_face_values.normal_vector(q_point)*/
                                                     topstress_values[q_point]*
                                                     rock_fe_face_values.shape_value(i,q_point) *
                                                     rock_fe_face_values.JxW(q_point));
                                }
                            }
                        }
                    
                    for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                        if (cell->face(face_number)->at_boundary()
                            &&
                            (cell->face(face_number)->boundary_id() == 2))
                        {
                            const unsigned int component_i = rock_fe.system_to_component_index(i).first;
                            rock_fe_face_values.reinit (cell, face_number);
                            
                            bottomstress.value_list (rock_fe_face_values.get_quadrature_points(),
                                                     bottomstress_values);
                            
                            for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                            {
                                for (unsigned int i=0; i<dofs_per_cell; ++i)
                                    local_rhs(i) += (bottomstress_values[q_point]*/*rock_fe_face_values.normal_vector(q_point)[component_i] **/
                                                     rock_fe_face_values.shape_value(i,q_point) *
                                                     rock_fe_face_values.JxW(q_point));
                            }
                        }
                    
                }
            }
            
            cell->get_dof_indices (local_dof_indices);
            rock_constraints.distribute_local_to_global (local_matrix, local_rhs,
                                                         local_dof_indices,
                                                         rock_system_matrix, rock_system_rhs);
        }
        
        // Need a condition as Dirichlet conditions as is makes the pressure only determined to a constant
        
        std::map<types::global_dof_index, double> pr_determination;
        {
            types::global_dof_index n_dofs = rock_dof_handler.n_dofs();
            std::vector<bool> componentVector(dim + 1, false); 
            componentVector[dim] = true;
            
            std::vector<bool> selected_dofs(n_dofs);
            std::set< types::boundary_id > boundary_ids;
            boundary_ids.insert(2);
            
            //			const RockBoundaryValues<dim> values;
            
            DoFTools::extract_boundary_dofs(rock_dof_handler, ComponentMask(componentVector),
                                            selected_dofs, boundary_ids);
            
            for (types::global_dof_index i = 0; i < n_dofs; i++) 
            {
                if (selected_dofs[i]) pr_determination[i] = data::pr_constant;
            }
        }
        MatrixTools::apply_boundary_values(pr_determination,
                                           rock_system_matrix, rock_solution, rock_system_rhs);
        
    }
    
    template <int dim>
    void DAE<dim>::solve_rock_system ()
    {
        SparseDirectUMFPACK  A_direct;
        A_direct.initialize(rock_system_matrix);
        A_direct.vmult (rock_solution, rock_system_rhs);
        rock_constraints.distribute (rock_solution);
    }
    
    
    
    template <int dim>
    void
    DAE<dim>::output_results ()
    {
        std::vector<std::string> solution_names (dim, "velocity");
        solution_names.push_back ("pressure");
        
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation
        (dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation
        .push_back (DataComponentInterpretation::component_is_scalar);
        
        DataOut<dim> data_out;
        data_out.attach_dof_handler (rock_dof_handler);
        data_out.add_data_vector (rock_solution, solution_names,
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
    DAE<dim>::compute_errors ()
	{
        {
            const ComponentSelectFunction<dim> pressure_mask (dim, dim+1);
            const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);
            RockExactSolution<dim> exact_solution;
            
            Vector<double> cellwise_errors (triangulation.n_active_cells());
            
            QTrapez<1>     q_trapez;
            QIterated<dim> quadrature (q_trapez, pr_degree+2);
            
            VectorTools::integrate_difference (rock_dof_handler, rock_solution, exact_solution,
                                               cellwise_errors, quadrature,
                                               VectorTools::L2_norm,
                                               &pressure_mask);
            const double p_l2_error = cellwise_errors.l2_norm();
            
            VectorTools::integrate_difference (rock_dof_handler, rock_solution, exact_solution,
                                               cellwise_errors, quadrature,
                                               VectorTools::L2_norm,
                                               &velocity_mask);
            const double u_l2_error = cellwise_errors.l2_norm();
            std::cout << "Errors: ||e_pr||_L2  = " << p_l2_error
            << ",  " << std::endl << "        ||e_vr||_L2  = " << u_l2_error
            << std::endl;
        }
    }
    
    
    
    template <int dim>
    void DAE<dim>::run ()
    {
            timestep_number = 0;
            time            = 0;
            make_grid ();
            setup_rock_dofs ();
            setup_phi_dofs ();
        
            VectorTools::interpolate(phi_dof_handler, PhiInitialFunction<dim>(), old_phi_solution);
//            VectorTools::interpolate(vf_dof_handler, vfInitialFunction<dim>(), vf_initial_solution);
//            VectorTools::interpolate(T_dof_handler, TempInitialFunction<dim>(), old_T_solution);
        
            phi_solution = old_phi_solution;
//            T_solution = old_T_solution;
        
            std::cout << "   Assembling..." << std::endl << std::flush;
            assemble_rock_system ();
        
            std::cout << "   Problem Degree = " << data::problem_degree << ". " << std::endl;
            std::cout << "   Refinement level = " << data::refinement_level << ". " << std::endl;
            		std::cout << "   Assembling..." << std::endl;
            std::cout << "   Solving..." << std::flush;
            solve_rock_system ();
            
            output_results ();
            compute_errors ();
            
            std::cout << std::endl;
    }
}

int main ()
{
    try
    {
        using namespace dealii;
        using namespace Step22;
        
        DAE<data::dimension> flow_problem(data::problem_degree);
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
