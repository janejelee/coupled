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
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_raviart_thomas.h>
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
        const unsigned int degree_phi = 1;
        
        const int refinement_level = 3;
        const double top = 1.0;
        const double bottom = 0.0;
        const double left = 0.0;
        const double right = PI;
        
        const double lambda = 1.0;
        const double perm_const = 1.0;
        const double rho_f = 1.0;
        const double phi = 0.7;
        const double pr_constant = (1-phi)*bottom*bottom*bottom;
        const double C = 1;
        
        const double timestep = 0.2;
        const double initial_time = 0.0;
        unsigned int timestep_number = 0;
        unsigned int total_timesteps = 1;
        double present_time;
    }
    
    using namespace data;
    
    template <int dim>
    class FullMovingMesh
    {
    public:
        FullMovingMesh (const unsigned int degree_rock, const unsigned int degree_pf, const unsigned int degree_vf);
        ~FullMovingMesh ();
        void run ();
    private:
        void make_initial_grid ();
        void setup_dofs_rock ();
        void setup_dofs_fluid ();
        void setup_dofs_phi ();
        void apply_BC_rock ();
        void assemble_system_rock ();
        void assemble_system_pf ();
        void assemble_system_vf ();
        void assemble_system_phi ();
        void assemble_rhs_phi ();
        void solve_rock ();
        void solve_fluid ();
        void solve_phi ();
        void output_results () const;
        void move_mesh ();
        void print_mesh ();
        void compute_errors ();
        
        const unsigned int        degree_rock;
        Triangulation<dim>        triangulation;
        FESystem<dim>             fe_rock;
        DoFHandler<dim>           dof_handler_rock;
        ConstraintMatrix          constraints_rock;
        BlockSparsityPattern      sparsity_pattern_rock;
        BlockSparseMatrix<double> system_matrix_rock;
        BlockVector<double>       solution_rock;
        BlockVector<double>       system_rhs_rock;
        
        const unsigned int        degree_pf;
        FESystem<dim>             fe_pf;
        DoFHandler<dim>           dof_handler_pf;
        ConstraintMatrix          constraints_pf;
        BlockSparsityPattern      sparsity_pattern_pf;
        BlockSparseMatrix<double> system_matrix_pf;
        BlockVector<double>       solution_pf;
        BlockVector<double>       system_rhs_pf;
        
        const unsigned int        degree_vf;
        FESystem<dim>             fe_vf;
        DoFHandler<dim>           dof_handler_vf;
        ConstraintMatrix          constraints_vf;
        SparsityPattern			  sparsity_pattern_vf;
        SparseMatrix<double>	  system_matrix_vf;
        SparseMatrix<double> 	  mass_matrix_vf;
        Vector<double>			  solution_vf;
        Vector<double>			  system_rhs_vf;
        
        DoFHandler<dim>           dof_handler_phi;
        FE_Q<dim>                 fe_phi;
        ConstraintMatrix          hanging_node_constraints_phi;
        SparsityPattern           sparsity_pattern_phi;
        SparseMatrix<double>      system_matrix_phi;
        SparseMatrix<double>      mass_matrix_phi;
        SparseMatrix<double>      nontime_matrix_phi;
        Vector<double>            solution_phi;
        Vector<double>            old_solution_phi;
        Vector<double>            system_rhs_phi;
        Vector<double>            nontime_rhs_phi;
    
        Vector<double>            displacement;
        
    };
  
    template <int dim>
    class ExactSolution_rock : public Function<dim>
    {
    public:
        ExactSolution_rock () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    
    template <int dim>
    class ExactSolution_pf : public Function<dim>
    {
    public:
        ExactSolution_pf () : Function<dim>(dim+1) {}
        
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    
    template <int dim>
    class ExactSolution_vf : public Function<dim>
    {
    public:
        ExactSolution_vf () : Function<dim>(dim+1) {}
        
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };

    template <int dim>
    void ExactSolution_rock<dim>::vector_value (const Point<dim> &p,
                                          Vector<double>   &values) const
    {
        values(0) = 0;
        values(1) = -(1-phi)*p[1]*p[1];
        values(2) = (1-phi)*p[1]*p[1]*p[1];
    }
    
    template <int dim>
    void ExactSolution_pf<dim>::vector_value (const Point<dim> &p,
                                             Vector<double>   &values) const
    {
            Assert (values.size() == dim+1,
                    ExcDimensionMismatch (values.size(), dim+1));
            
            const double permeability = perm_const;
            
            values(0) = 0.0;
            values(1) = lambda*rho_f*(1-p[1]*p[1])*permeability;
            values(2) = -rho_f*(p[1] - (1.0/3.0)*p[1]*p[1]*p[1]);
    }
    
    template <int dim>
    void ExactSolution_vf<dim>::vector_value (const Point<dim> &p,
                                             Vector<double>   &values) const
	{
            values(0) = 0.0;
            values(1) = (-rho_f*(1-p[1]*p[1])) ;
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
    class ExtraRHSpf : public Function<dim>
    {
    public:
        ExtraRHSpf () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double ExtraRHSpf<dim>::value (const Point<dim>   &p,
                                      const unsigned int) const
    {
        return 2*p[1]*(1-phi+lambda*perm_const*rho_f);
    }
    
    template <int dim>
    class PressureBoundaryValues : public Function<dim>
    {
    public:
        PressureBoundaryValues () : Function<dim>(1) {}
        
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>    
    double PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                                   const unsigned int /*component*/) const
        {
            // This is the dirichlet condition for the top
            return -rho_f*(p[1]-1./3*p[1]*p[1]*p[1]); // last term added to avoid warnings on work comp
        }
    
    // Use this if you know where the inflow boundaries are and can define them
    template <int dim>
    class PhiDiriFunction : public Function<dim>
    {
    public:
        PhiDiriFunction () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double PhiDiriFunction<dim>::value (const Point<dim>  &p,
                                     const unsigned int /*component*/) const
    {
        const double time = this->get_time();
        return  C * time * p[0]*exp(-p[1]);
    }
    
    template <int dim>
    class KInverse : public TensorFunction<2,dim>
    {
    public:
        KInverse () : TensorFunction<2,dim>() {}
        
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<Tensor<2,dim> >    &values) const;
    };
    
    template <int dim>
    void
    KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                               std::vector<Tensor<2,dim> >    &values) const
    {
        
        Assert (points.size() == values.size(),
                ExcDimensionMismatch (points.size(), values.size()));
        
        for (unsigned int p=0; p<points.size(); ++p)
        {
            values[p].clear ();
            const double permeability = perm_const;
            
            for (unsigned int d=0; d<dim; ++d)
                values[p][d][d] = 1./permeability;
        }
    }
    
    template <int dim>
    class K : public TensorFunction<2,dim>
    {
    public:
        K () : TensorFunction<2,dim>() {}
        
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<Tensor<2,dim> >    &values) const;
    };
    
    template <int dim>
    void
    K<dim>::value_list (const std::vector<Point<dim> > &points,
                        std::vector<Tensor<2,dim> >    &values) const
    {
        Assert (points.size() == values.size(),
                ExcDimensionMismatch (points.size(), values.size()));
        
        for (unsigned int p=0; p<points.size(); ++p)
        {
            values[p].clear ();
            Assert (points.size() == values.size(),
                    ExcDimensionMismatch (points.size(), values.size()));
            
            for (unsigned int p=0; p<points.size(); ++p)
            {
                values[p].clear ();/*
                                    const double distance_to_flowline
                                    = std::fabs(points[p][1]-0.2*std::sin(10*points[p][0]));*/
                const double permeability = perm_const;
                
                for (unsigned int d=0; d<dim; ++d)
                    values[p][d][d] = permeability;
            }
        }
    }
        
    template <int dim>
    void unitz (const std::vector<Point<dim> > &points,
                    std::vector<Tensor<1, dim> >   &values)
        {
            for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
            {
                values[point_n][0] = 0.0;
                values[point_n][1] = 1.0;
            }
        }
    
    template <int dim>
    FullMovingMesh<dim>::FullMovingMesh (const unsigned int degree_rock, const unsigned int degree_pf, const unsigned int degree_vf)
    :
    degree_rock (degree_rock),
	triangulation (Triangulation<dim>::maximum_smoothing),
    fe_rock (FE_Q<dim>(degree_rock+1), dim,
        FE_Q<dim>(degree_rock), 1),
    dof_handler_rock (triangulation),
	
	degree_pf (degree_pf),
    fe_pf (FE_RaviartThomas<dim>(degree_pf), 1,
           FE_DGQ<dim>(degree_pf), 1),
	dof_handler_pf (triangulation),
    
    degree_vf (degree_vf),
    fe_vf (FE_Q<dim>(degree_vf), dim),
    dof_handler_vf (triangulation),
    
    dof_handler_phi (triangulation),
    fe_phi (degree_phi)

    {}
    
    template <int dim>
    FullMovingMesh<dim>::~FullMovingMesh ()
    {
        dof_handler_phi.clear ();
    }
    
    template <int dim>
    void FullMovingMesh<dim>::make_initial_grid ()
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
        
           triangulation.refine_global (refinement_level);
	}
    
    template <int dim>
    void FullMovingMesh<dim>::setup_dofs_rock ()
    {
       
        system_matrix_rock.clear ();
        dof_handler_rock.distribute_dofs (fe_rock);
        DoFRenumbering::Cuthill_McKee (dof_handler_rock);
        std::vector<unsigned int> block_component (dim+1,0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise (dof_handler_rock, block_component);
        
        std::vector<types::global_dof_index> dofs_per_block (2);
        DoFTools::count_dofs_per_block (dof_handler_rock, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0],
        n_p = dofs_per_block[1];
        std::cout 
        << "	Refinement level: "
        << refinement_level
        << std::endl
		<< "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Number of degrees of freedom in rock problem: "
        << dof_handler_rock.n_dofs()
        << " (" << n_u << '+' << n_p << ')'
        << std::endl;
        
            BlockDynamicSparsityPattern dsp_rock (2,2);
            dsp_rock.block(0,0).reinit (n_u, n_u);
            dsp_rock.block(1,0).reinit (n_p, n_u);
            dsp_rock.block(0,1).reinit (n_u, n_p);
            dsp_rock.block(1,1).reinit (n_p, n_p);
            dsp_rock.collect_sizes();
            DoFTools::make_sparsity_pattern (dof_handler_rock, dsp_rock, constraints_rock, false);
            sparsity_pattern_rock.copy_from (dsp_rock);
        
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
    void FullMovingMesh<dim>::setup_dofs_fluid ()
	{
        dof_handler_pf.distribute_dofs (fe_pf);
        dof_handler_vf.distribute_dofs (fe_vf);

        // fluid motion numbering
        DoFRenumbering::component_wise (dof_handler_pf);     
        std::vector<types::global_dof_index> dofs_per_component (dim+1);       
        DoFTools::count_dofs_per_component (dof_handler_pf, dofs_per_component);      
        const unsigned int n_u = dofs_per_component[0],
        					n_pf = dofs_per_component[dim],
							n_vf = dof_handler_vf.n_dofs();

        std::cout << "	Number of degrees of freedom in fluid problem: "
		<< dof_handler_pf.n_dofs()            
		<< " (" << n_u << '+' << n_pf << '+' << n_vf << ')'            
		<< std::endl;

        BlockDynamicSparsityPattern dsp_pf(2, 2);                  
        dsp_pf.block(0, 0).reinit (n_u, n_u);
        dsp_pf.block(1, 0).reinit (n_pf, n_u);      	
        dsp_pf.block(0, 1).reinit (n_u, n_pf);
        dsp_pf.block(1, 1).reinit (n_pf, n_pf);
                    	
        dsp_pf.collect_sizes ();
        DoFTools::make_sparsity_pattern (dof_handler_pf, dsp_pf, constraints_pf, false);

        sparsity_pattern_pf.copy_from(dsp_pf);                    	
        system_matrix_pf.reinit (sparsity_pattern_pf);                
        solution_pf.reinit (2);        
        solution_pf.block(0).reinit (n_u);        
        solution_pf.block(1).reinit (n_pf);        
        solution_pf.collect_sizes ();
        
        system_rhs_pf.reinit (2);        
        system_rhs_pf.block(0).reinit (n_u);
        system_rhs_pf.block(1).reinit (n_pf);
        system_rhs_pf.collect_sizes ();

        system_matrix_vf.clear ();
        constraints_vf.clear ();           
        DoFTools::make_hanging_node_constraints (dof_handler_vf,
                                                         constraints_vf);

        constraints_vf.close();
        DynamicSparsityPattern dsp_vf(dof_handler_vf.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_vf,
                                                dsp_vf,
                                                constraints_vf,
                                                /*keep_constrained_dofs = */ true);
        sparsity_pattern_vf.copy_from(dsp_vf);
        mass_matrix_vf.reinit (sparsity_pattern_vf);
        system_matrix_vf.reinit (sparsity_pattern_vf);
        solution_vf.reinit (dof_handler_vf.n_dofs());
        system_rhs_vf.reinit (dof_handler_vf.n_dofs());

        MatrixCreator::create_mass_matrix(dof_handler_vf,
                                                  QGauss<dim>(fe_vf.degree+2),
                                                  mass_matrix_vf);
	}
    
    template <int dim>
    void FullMovingMesh<dim>::setup_dofs_phi ()
    {
        dof_handler_phi.distribute_dofs (fe_phi);
        hanging_node_constraints_phi.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler_phi,
                                                 hanging_node_constraints_phi);
        hanging_node_constraints_phi.close ();
        
        DynamicSparsityPattern dsp(dof_handler_phi.n_dofs(), dof_handler_phi.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_phi,
                                        dsp, hanging_node_constraints_phi, /*keep_constrained_dofs = */ true);
        sparsity_pattern_phi.copy_from (dsp);
        
        system_matrix_phi.reinit (sparsity_pattern_phi);
        mass_matrix_phi.reinit (sparsity_pattern_phi);
        nontime_matrix_phi.reinit (sparsity_pattern_phi);
        solution_phi.reinit (dof_handler_phi.n_dofs());
        old_solution_phi.reinit(dof_handler_phi.n_dofs());
        system_rhs_phi.reinit (dof_handler_phi.n_dofs());
        nontime_rhs_phi.reinit (dof_handler_phi.n_dofs());
    }
    

    template <int dim>
    void FullMovingMesh<dim>::apply_BC_rock ()
	{
    	constraints_rock.clear ();
    	    	            
    	FEValuesExtractors::Vector velocities(0);
    	FEValuesExtractors::Scalar pressure (dim);
    	    	            
    	DoFTools::make_hanging_node_constraints (dof_handler_rock,constraints_rock);
    	    	            
    	//            Testing if Dirichlet conditions work: use the following for top boundary 1 and bottom boundary id 2
    	    	            
    	VectorTools::interpolate_boundary_values (dof_handler_rock,
    			1,
				ExactSolution_rock<dim>(),
				constraints_rock,
				fe_rock.component_mask(velocities));
    	VectorTools::interpolate_boundary_values (dof_handler_rock,
    			2,
				ExactSolution_rock<dim>(),
				constraints_rock,
				fe_rock.component_mask(velocities));
    	    	            
    	// These are the conditions for the side boundary ids 0 (no flux)
    	std::set<types::boundary_id> no_normal_flux_boundaries;
    	no_normal_flux_boundaries.insert (0);
    	VectorTools::compute_no_normal_flux_constraints (dof_handler_rock, 0,
    			no_normal_flux_boundaries, constraints_rock);
    	
    	constraints_rock.close ();
	}

    
    template <int dim>
    void FullMovingMesh<dim>::assemble_system_rock ()
    {
        system_matrix_rock=0;
        system_rhs_rock=0;
        QGauss<dim>   quadrature_formula(degree_rock+2);
        
        FEValues<dim> fe_values_rock (fe_rock, quadrature_formula,
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
            fe_values_rock.reinit (cell);
            local_matrix = 0;
            local_rhs = 0;
            right_hand_side.vector_value_list(fe_values_rock.get_quadrature_points(),
                                              rhs_values);
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
                        local_matrix(i,j) += (2 * (1-phi) *
                                              (symgrad_phi_u[i] * symgrad_phi_u[j])
                                              - (1-phi)*div_phi_u[i] * phi_p[j]
                                              - (1-phi)*phi_p[i] * div_phi_u[j])
                                                    * fe_values_rock.JxW(q);
                    }
                    
                    const unsigned int component_i =
                    fe_rock.system_to_component_index(i).first;
                    
                    // Add the extra terms on RHS
                    if (timestep_number==0)
                    {
                    local_rhs(i) +=  fe_values_rock.shape_value(i,q) *
                    (1- phi)*rhs_values[q](component_i) *
                                                        fe_values_rock.JxW(q);
                    }
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
    void FullMovingMesh<dim>::assemble_system_pf ()
    {
        QGauss<dim>   quadrature_formula(degree_pf+2);
        QGauss<dim-1> face_quadrature_formula(degree_pf+2);
        
        FEValues<dim> fe_values_pf (fe_pf, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        
        FEFaceValues<dim> fe_face_values_pf (fe_pf, face_quadrature_formula,
                                             update_values    | update_normal_vectors |
                                             update_quadrature_points  | update_JxW_values);

        FEValues<dim> fe_values_rock (fe_rock, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        
        const unsigned int   dofs_per_cell   = fe_pf.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        const ExtraRHSpf<dim>             pf_rhs_values;
        const PressureBoundaryValues<dim> pressure_boundary_values;
        const KInverse<dim>               k_inverse;
        const K<dim>               		  k;
        
        std::vector<double> div_vr_values (n_q_points);
        std::vector<double> extra_pf_rhs (n_q_points);
        std::vector<double> boundary_values (n_face_q_points);
        std::vector<Tensor<2,dim> > k_inverse_values (n_q_points);
        std::vector<Tensor<2,dim> > k_values (n_q_points);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_pf.begin_active(),
        endc = dof_handler_pf.end();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = dof_handler_rock.begin_active();
        for (; cell!=endc; ++cell, ++vr_cell)
        {
            fe_values_pf.reinit (cell);
            fe_values_rock.reinit(vr_cell);

            local_matrix = 0;
            local_rhs = 0;
            
            k_inverse.value_list (fe_values_pf.get_quadrature_points(),
                                  k_inverse_values);
            k.value_list (fe_values_pf.get_quadrature_points(),
                          k_values);
            fe_values_rock[velocities].get_function_divergences (solution_rock, div_vr_values);
            pf_rhs_values.value_list (fe_values_pf.get_quadrature_points(), extra_pf_rhs);

            for (unsigned int q=0; q<n_q_points; ++q)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    const Tensor<1,dim> phi_i_u     = fe_values_pf[velocities].value (i, q);
                    const double        div_phi_i_u = fe_values_pf[velocities].divergence (i, q);
                    const double        phi_i_p     = fe_values_pf[pressure].value (i, q);
                    
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                        const Tensor<1,dim> phi_j_u     = fe_values_pf[velocities].value (j, q);
                        const double        div_phi_j_u = fe_values_pf[velocities].divergence (j, q);
                        const double        phi_j_p     = fe_values_pf[pressure].value (j, q);
                        
                        local_matrix(i,j) += ( 1./lambda * phi_i_u * k_inverse_values[q] * phi_j_u
                                              - div_phi_i_u * phi_j_p
                                              - phi_i_p * div_phi_j_u)
											  * fe_values_pf.JxW(q);
                    }
                    
                    local_rhs(i) += phi_i_p *
                    					(div_vr_values[q] + extra_pf_rhs[q]) *
										fe_values_pf.JxW(q);
                    // BE CAREFUL HERE ONCE K is not constant or 1 anymore. have taken div of the constants as zero
                }
            
            for (unsigned int face_no=0;
                 face_no<GeometryInfo<dim>::faces_per_cell;
                 ++face_no)
                if (cell->face(face_no)->at_boundary()
                    &&
                    (cell->face(face_no)->boundary_id() == 1)) // Basin top has boundary id 1
                {
                    fe_face_values_pf.reinit (cell, face_no);
                    // DIRICHLET CONDITION FOR TOP. pf = pr at top of basin
                    
                    pressure_boundary_values
                    .value_list (fe_face_values_pf.get_quadrature_points(),
                                 boundary_values);
                    
                    for (unsigned int q=0; q<n_face_q_points; ++q)
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                            local_rhs(i) += -( fe_face_values_pf[velocities].value (i, q) *
                                              fe_face_values_pf.normal_vector(q) *
                                              boundary_values[q] *
                                              fe_face_values_pf.JxW(q));
                }
            
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    system_matrix_pf.add (local_dof_indices[i],
                                          local_dof_indices[j],
                                          local_matrix(i,j));
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                system_rhs_pf(local_dof_indices[i]) += local_rhs(i);
        }
        
        // NOW NEED TO STRONGLY IMPOSE THE FLUX CONDITIONS BOTH ON SIDES AND BOTTOM
        {std::map<types::global_dof_index, double> boundary_values_flux;
        {
            types::global_dof_index n_dofs = dof_handler_pf.n_dofs();
            std::vector<bool> componentVector(dim + 1, true); // condition is on pressue
            // setting flux value for the sides at 0 ON THE PRESSURE
            componentVector[dim] = false;
            std::vector<bool> selected_dofs(n_dofs);
            std::set< types::boundary_id > boundary_ids;
            boundary_ids.insert(0);

            DoFTools::extract_boundary_dofs(dof_handler_pf, ComponentMask(componentVector),
                                            selected_dofs, boundary_ids);

            for (types::global_dof_index i = 0; i < n_dofs; i++) {
                if (selected_dofs[i]) boundary_values_flux[i] = 0.0; // Side boudaries have flux 0 on pressure
            }
        }
            // Apply the conditions
            MatrixTools::apply_boundary_values(boundary_values_flux,
                                               system_matrix_pf, solution_pf, system_rhs_pf);
        }
//        {
//            std::map<types::global_dof_index, double> boundary_values_flux;
//            {
//                types::global_dof_index n_dofs = dof_handler_pf.n_dofs();
//                std::vector<bool> componentVector(dim + 1, true); // condition is on pressue
//                // setting flux value for the sides at 0 ON THE PRESSURE
//                componentVector[dim] = false;
//                std::vector<bool> selected_dofs(n_dofs);
//                std::set< types::boundary_id > boundary_ids;
//                boundary_ids.insert(2);
//
//                DoFTools::extract_boundary_dofs(dof_handler_pf, ComponentMask(componentVector),
//                                                selected_dofs, boundary_ids);
//
//                for (types::global_dof_index i = 0; i < n_dofs; i++) {
//                    if (selected_dofs[i]) boundary_values_flux[i] = -rho_f; // Side boudaries have flux 0 on pressure
//                }
//            }
//            // Apply the conditions
//            MatrixTools::apply_boundary_values(boundary_values_flux,
//                                               system_matrix_pf, solution_pf, system_rhs_pf);
//        }
        
    }
    
    template <int dim>
    void FullMovingMesh<dim>::assemble_system_vf ()
    {
        QGauss<dim>   quadrature_formula(degree_vf+2);
        FEValues<dim> fe_values_vf (fe_vf, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        FEValues<dim> fe_values_pf (fe_pf, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        FEValues<dim> fe_values_rock (fe_rock, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
	       
        const unsigned int   dofs_per_cell   = fe_vf.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
	       
        std::vector<Tensor<1,dim>>     grad_pf_values (n_q_points);
        std::vector<Tensor<1,dim>>     unitz_values (n_q_points);
        std::vector<Tensor<1,dim>> 	  vr_values (n_q_points);
        std::vector<double>             pf_values (n_q_points);
	       
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_vf.begin_active(),
        endc = dof_handler_vf.end();
        typename DoFHandler<dim>::active_cell_iterator
        pf_cell = dof_handler_pf.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = dof_handler_rock.begin_active();
        for (; cell!=endc; ++cell, ++pf_cell, ++vr_cell)
	     	 {
                 fe_values_vf.reinit(cell);
                 fe_values_pf.reinit(pf_cell);
                 fe_values_rock.reinit(vr_cell);
                 
                 local_matrix = 0;
                 local_rhs = 0;
                 
                 unitz (fe_values_vf.get_quadrature_points(), unitz_values);
                 fe_values_pf[pressure].get_function_gradients (solution_pf, grad_pf_values);
                 fe_values_rock[velocities].get_function_values (solution_rock, vr_values);
                 fe_values_pf[pressure].get_function_values (solution_pf, pf_values);
                 
                 for (unsigned int i=0; i<dofs_per_cell; ++i)
                 {
                     const unsigned int
                     component_i = fe_vf.system_to_component_index(i).first;
                     
                     for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                     { local_rhs(i) +=    (
                                             grad_pf_values[q_point][component_i]
                                             ) * fe_values_vf.shape_value(i,q_point) *
											 	 	 fe_values_vf.JxW(q_point);
                     }
                 }
                 
                 cell->get_dof_indices (local_dof_indices);
                 for (unsigned int i=0; i<dofs_per_cell; ++i)
                 {system_rhs_vf(local_dof_indices[i]) += local_rhs(i);
                 }
             }
        
        constraints_vf.condense (system_matrix_vf);
        constraints_vf.condense (system_rhs_vf);
        system_matrix_vf.copy_from(mass_matrix_vf);
        
    }
    
    template <int dim>
    void FullMovingMesh<dim>::assemble_system_phi ()
    {
        
        system_matrix_phi = 0;
        nontime_matrix_phi = 0;
        
        MatrixCreator::create_mass_matrix(dof_handler_phi,
                                          QGauss<dim>(degree_phi+2),
                                          mass_matrix_phi);
        
        FullMatrix<double>                   cell_matrix;
        
        std::vector<types::global_dof_index> local_dof_indices;
        
        QGauss<dim>  quadrature_formula(degree_phi);
        QGauss<dim-1> face_quadrature_formula(degree_phi);
        
        FEValues<dim> fe_values_phi (fe_phi, quadrature_formula,
                                     update_values    | update_gradients |
                                     update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values_phi (fe_phi, face_quadrature_formula,
                                          update_values | update_quadrature_points |
                                          update_JxW_values | update_normal_vectors);
        FEValues<dim> fe_values_vr (fe_rock, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);

        const unsigned int dofs_per_cell   = fe_phi.dofs_per_cell;
        const unsigned int n_q_points      = fe_values_phi.get_quadrature().size();
//        const unsigned int n_face_q_points = fe_face_values_phi.get_quadrature().size();
        
        cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        
        std::vector<Tensor<1,dim>>       vr_values (n_q_points);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_phi.begin_active(),
        endc = dof_handler_phi.end();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = dof_handler_rock.begin_active();
        for (; cell!=endc; ++cell, ++vr_cell)
        {
            cell_matrix = 0;
            
            fe_values_phi.reinit(cell);
            fe_values_vr.reinit(vr_cell);

            fe_values_vr[velocities].get_function_values (solution_rock, vr_values);
            
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    // (v_r.grad phi, psi+d*v_r.grad_psi)
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        cell_matrix(i,j) += ((vr_values[q_point] *
                                              fe_values_phi.shape_grad(j,q_point)   *
                                              (fe_values_phi.shape_value(i,q_point)
                                               //                                               +
                                               //                                               delta *
                                               //                                               (advection_directions[q_point] *
                                               //                                                fe_values.shape_grad(i,q_point))
                                               )) *
                                             fe_values_phi.JxW(q_point));
                }
            
            cell->get_dof_indices (local_dof_indices);
            
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                    nontime_matrix_phi.add (local_dof_indices[i],
                                        local_dof_indices[j],
                                        cell_matrix(i,j));
            }
        }
    }
    
    template <int dim>
    void FullMovingMesh<dim>::assemble_rhs_phi ()
    {
        system_rhs_phi=0;
        nontime_rhs_phi=0;
        
        Vector<double>                       cell_rhs;
        Vector<double>                       cell_disp;
        std::vector<types::global_dof_index> local_dof_indices;
        
        QGauss<dim>  quadrature_formula(degree_phi+2);
        QGauss<dim-1> face_quadrature_formula(degree_phi+2);
        
        FEValues<dim> fe_values_phi (fe_phi, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> fe_face_values_phi (fe_phi, face_quadrature_formula,
                                          update_values | update_quadrature_points |
                                          update_JxW_values | update_normal_vectors);
        FEValues<dim> fe_values_pf (fe_pf, quadrature_formula,
                                    update_values    |  update_gradients |
                                    update_quadrature_points  |  update_JxW_values);
        FEValues<dim> fe_values_rock (fe_rock, quadrature_formula,
                                    update_values    |  update_gradients |
                                    update_quadrature_points  |  update_JxW_values);
 
        const unsigned int dofs_per_cell   = fe_phi.dofs_per_cell;
        const unsigned int n_q_points      = fe_values_phi.get_quadrature().size();
//        const unsigned int n_face_q_points = fe_face_values_phi.get_quadrature().size();
        
        cell_rhs.reinit (dofs_per_cell);
        cell_disp.reinit (dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        std::vector<double>              pr_values (n_q_points);
        std::vector<double>              pf_values (n_q_points);

        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_phi.begin_active(),
        endc = dof_handler_phi.end();
        typename DoFHandler<dim>::active_cell_iterator
        rock_cell = dof_handler_rock.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        pf_cell = dof_handler_pf.begin_active();
        for (; cell!=endc; ++cell, ++rock_cell, ++pf_cell)
        {
            cell_rhs = 0;
            
            fe_values_phi.reinit (cell);
            fe_values_pf.reinit (pf_cell);
            fe_values_rock.reinit (rock_cell);
            
            fe_values_pf[pressure].get_function_values (solution_pf, pf_values);
            fe_values_rock[pressure].get_function_values (solution_rock, pr_values);

            
            //            const double delta = 0.1 * cell->diameter ();
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    cell_rhs(i) += ((fe_values_phi.shape_value(i,q_point)
                                     //                                     +
                                     //                                     delta *
                                     //                                     (advection_directions[q_point] *
                                     //                                      fe_values.shape_grad(i,q_point))
                                     ) *
                                    (pr_values[q_point]-pf_values[q_point]) *
                                    fe_values_phi.JxW (q_point));
                }
            
            cell->get_dof_indices (local_dof_indices);
            
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                nontime_rhs_phi(local_dof_indices[i]) += cell_rhs(i);
            }
        }
    }
    
    template <int dim>
    void FullMovingMesh<dim>::solve_rock ()
    {
        std::cout << "   Solving for rock system..." << std::endl;
        SparseDirectUMFPACK  A_direct;
        A_direct.initialize(system_matrix_rock);
        A_direct.vmult (solution_rock, system_rhs_rock);
        constraints_rock.distribute (solution_rock);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::solve_fluid ()
    {
        std::cout << "   Solving for p_f..." << std::endl;
        SparseDirectUMFPACK  pf_direct;
        pf_direct.initialize(system_matrix_pf);
        pf_direct.vmult (solution_pf, system_rhs_pf);
    
        assemble_system_vf ();
        std::cout << "   Solving for v_f..." << std::endl;
        SparseDirectUMFPACK  vf_direct;
        vf_direct.initialize(system_matrix_vf);
        vf_direct.vmult (solution_vf, system_rhs_vf);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::solve_phi ()
    {
        mass_matrix_phi.vmult(system_rhs_phi, old_solution_phi);
        system_rhs_phi.add(timestep,nontime_rhs_phi);
        
        system_matrix_phi.copy_from(mass_matrix_phi);
        system_matrix_phi.add(timestep,nontime_matrix_phi);
        
        std::map<types::global_dof_index,double> boundary_values;
        PhiDiriFunction<dim> phi_boundary_values;
        phi_boundary_values.set_time(present_time);
        VectorTools::interpolate_boundary_values (dof_handler_phi,
                                                  1,
                                                  phi_boundary_values,
                                                  boundary_values);
        MatrixTools::apply_boundary_values (boundary_values,
                                            system_matrix_phi,
                                            solution_phi,
                                            system_rhs_phi);
        
        hanging_node_constraints_phi.condense (system_rhs_phi);
        hanging_node_constraints_phi.condense (system_matrix_phi);
        
        SparseDirectUMFPACK  phi_direct;
        phi_direct.initialize(system_matrix_phi);
        phi_direct.vmult (solution_phi, system_rhs_phi);
        
        hanging_node_constraints_phi.distribute (solution_phi);
    }
    
    template <int dim>
    void
    FullMovingMesh<dim>::output_results ()  const
    {
        // ROCK
        std::vector<std::string> solution_names (dim, "velocity");
        solution_names.push_back ("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation
        (dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation
        .push_back (DataComponentInterpretation::component_is_scalar);
        DataOut<dim> data_out_rock;
        data_out_rock.attach_dof_handler (dof_handler_rock);
        data_out_rock.add_data_vector (solution_rock, solution_names,
                                  DataOut<dim>::type_dof_data,
                                  data_component_interpretation);
        data_out_rock.build_patches ();
        std::ostringstream filename_rock;
        filename_rock << "solution_rock"+ Utilities::int_to_string(timestep_number, 3) + ".vtk";
        std::ofstream output_rock (filename_rock.str().c_str());
        data_out_rock.write_vtk (output_rock);
        
        //FLUID
        std::vector<std::string> pf_names (dim, "uf");
        pf_names.push_back ("p_f");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        pf_component_interpretation
        (dim+1, DataComponentInterpretation::component_is_scalar);
        for (unsigned int i=0; i<dim; ++i)
            pf_component_interpretation[i]
            = DataComponentInterpretation::component_is_part_of_vector;
        DataOut<dim> data_out_fluid;
        
        data_out_fluid.add_data_vector (dof_handler_pf, solution_pf,
                                  pf_names, pf_component_interpretation);
        data_out_fluid.add_data_vector (dof_handler_vf, solution_vf,
                                  "v_f");
        
        data_out_fluid.build_patches ();
        const std::string filename_fluid = "solution_fluid-"
        + Utilities::int_to_string(timestep_number, 3) +
        ".vtk";
        std::ofstream output_fluid (filename_fluid.c_str());
        data_out_fluid.write_vtk (output_fluid);
        
        //PHI
        DataOut<dim> data_out_phi;

        data_out_phi.attach_dof_handler(dof_handler_phi);
        data_out_phi.add_data_vector(solution_phi, "phi");
        data_out_phi.build_patches();
        
        const std::string filename_phi = "solution_phi-"
        + Utilities::int_to_string(timestep_number, 3) +
        ".vtk";
        std::ofstream output_phi(filename_phi.c_str());
        data_out_phi.write_vtk(output_phi);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::move_mesh ()
    {
        std::cout << "   Moving mesh..." << std::endl;
        
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
    void FullMovingMesh<dim>::print_mesh ()
    {
            std::ofstream out ("grid-"
                               + Utilities::int_to_string(timestep_number, 3) +
                               ".eps");
            GridOut grid_out;
            grid_out.write_eps (triangulation, out);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::compute_errors ()
    {
        {
            const ComponentSelectFunction<dim> pressure_mask (dim, dim+1);
            const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);
            ExactSolution_rock<dim> exact_solution_rock;
            
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
        {
            const ComponentSelectFunction<dim> pressure_mask (dim, dim+1);
            const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);
            ExactSolution_pf<dim> exact_solution_pf;
            
            Vector<double> cellwise_errors (triangulation.n_active_cells());
            
            QTrapez<1>     q_trapez;
            QIterated<dim> quadrature (q_trapez, degree_pf+2);
            
            VectorTools::integrate_difference (dof_handler_pf, solution_pf, exact_solution_pf,
                                               cellwise_errors, quadrature,
                                               VectorTools::L2_norm,
                                               &pressure_mask);
            const double pf_l2_error = cellwise_errors.l2_norm();
            
            VectorTools::integrate_difference (dof_handler_pf, solution_pf, exact_solution_pf,
                                               cellwise_errors, quadrature,
                                               VectorTools::L2_norm,
                                               &velocity_mask);
            const double u_l2_error = cellwise_errors.l2_norm();
            std::cout << "   Errors: ||e_pf||_L2  = " << pf_l2_error
            << ",  " << std::endl << "           ||e_u||_L2  = " << u_l2_error
            << std::endl;
        }
        {
            ExactSolution_vf<dim> exact_solution_vf;
            
            Vector<float> difference_per_cell (triangulation.n_active_cells());
            VectorTools::integrate_difference (dof_handler_vf,
                                               solution_vf,
                                               exact_solution_vf,
                                               difference_per_cell,
                                               QGauss<dim>(3),
                                               VectorTools::L2_norm);
            const double vf_l2_error = difference_per_cell.l2_norm();
            std::cout << "   Errors: ||e_vf||_L2  = " << vf_l2_error
            << ",  " << std::endl;
        }
    }
    
    template <int dim>
    void FullMovingMesh<dim>::run ()
    {
        present_time = initial_time;
        
        make_initial_grid ();
        setup_dofs_rock ();
        setup_dofs_fluid ();
        setup_dofs_phi ();

        while (timestep_number < total_timesteps)
        {
        	apply_BC_rock ();
            std::cout << "   Assembling at timestep number "
                            << timestep_number << "..." <<  std::endl << std::flush;
            assemble_system_rock ();
            solve_rock ();
            
            assemble_system_pf ();
            solve_fluid ();
            
            assemble_system_phi ();
            assemble_rhs_phi ();
            solve_phi ();
            
            output_results ();
            compute_errors ();
            
            print_mesh ();
            move_mesh ();
            
            ++timestep_number;
            present_time += timestep;
        }
    }
}

int main ()
{
    try
    {
        using namespace dealii;
        using namespace FullSolver;
        FullMovingMesh<2> flow_problem(1, 1, 2);
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
