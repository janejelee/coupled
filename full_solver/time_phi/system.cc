

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

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/base/tensor_function.h>

namespace System
{
    using namespace dealii;
    using namespace numbers;
    namespace data
    {
    const double eta = 1.0;
    const double lambda = 1.0;
    const double perm_const = 1.0;
    const double phi = 0.7;
    const double rho_f = 1.0;
    const double rho_r = 1.0;
    const double gamma = 1.0;

    const double top = 1.0;
    const double bottom = 0.0;
    const double left = 0.0;
    const double right = PI;

    const int dimension = 2;
    const int problem_degree = 1;
    const int refinement_level = 4;

        const double timestep = 1e-6;
    const double final_time = 1*timestep;
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
        void setup_fluid_dofs ();
        void setup_rock_dofs ();
        void setup_phi_dofs ();

        void assemble_rock_system ();
        void solve_rock_system ();
        void assemble_pf_system ();
        void assemble_vf_system ();
        void solve_fluid_system ();

        void assemble_phi_system ();
        void assemble_phi_rhs ();
        void solve_phi ();

        void compute_errors () const;
        void output_results () const;
        void output_phi_results ();
        Triangulation<dim>   triangulation;

        const unsigned int   pr_degree;
        FESystem<dim>        rock_fe;
        DoFHandler<dim>      rock_dof_handler;
        ConstraintMatrix     rock_constraints;

        BlockSparsityPattern     	rock_sparsity_pattern;
        BlockSparseMatrix<double> 	rock_system_matrix;
        BlockVector<double> 		rock_solution;
        BlockVector<double> 		rock_system_rhs;

        const unsigned int   pf_degree;
        FESystem<dim>        pf_fe;
        DoFHandler<dim>      pf_dof_handler;
        ConstraintMatrix     pf_constraints;

        BlockSparsityPattern      pf_sparsity_pattern;
        BlockSparseMatrix<double> pf_system_matrix;
        BlockVector<double>       pf_solution;
        BlockVector<double>       pf_initial_solution;
        BlockVector<double>       pf_system_rhs;

        const unsigned int 	vf_degree;
        FESystem<dim>		vf_fe;
        DoFHandler<dim>     vf_dof_handler;
        ConstraintMatrix    vf_constraints;

        SparsityPattern      vf_sparsity_pattern;
        SparseMatrix<double> vf_system_matrix;
        SparseMatrix<double> vf_mass_matrix;
        Vector<double>       vf_solution;
        Vector<double>       vf_initial_solution;
        Vector<double>       vf_system_rhs;

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

        Vector<double>       phi_pf;

        double               time;
        double               time_step;
        unsigned int         timestep_number;

    };



    template <int dim>
    class RockBoundaryValuesTop : public Function<dim>
    {
    public:
      RockBoundaryValuesTop () : Function<dim>(dim+1) {}
      virtual double value (const Point<dim>   &p,
                            const unsigned int  component = 0) const;
      virtual void vector_value (const Point<dim> &p,
                                 Vector<double>   &value) const;
    };
    template <int dim>
    double
    RockBoundaryValuesTop<dim>::value (const Point<dim>  &p,
                                const unsigned int component) const
    {

    		if (component == 0) // conditions for top and bottome
    			return std::sin(p[0]);
    		else if (component == 1)
    			return -p[0]*p[1]*p[1];
    		else if (component == dim)
        			return p[0]*p[1];
      return 0.0;
    }

    template <int dim>
    void
    RockBoundaryValuesTop<dim>::vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = RockBoundaryValuesTop<dim>::value (p, c);
    }


    template <int dim>
    class RockBoundaryValuesBottom : public Function<dim>
    {
    public:
      RockBoundaryValuesBottom () : Function<dim>(dim+1) {}
      virtual double value (const Point<dim>   &p,
                            const unsigned int  component = 0) const;
      virtual void vector_value (const Point<dim> &p,
                                 Vector<double>   &value) const;
    };
    template <int dim>
    double
    RockBoundaryValuesBottom<dim>::value (const Point<dim>  &p,
                                const unsigned int component) const
    {

    		if (component == 0)
    			return p[0]*0.0; // put in only to omit warning on work computer
    		else if (component == 1)
    			return 0.0;
      return 0.0;
    }

    template <int dim>
    void
    RockBoundaryValuesBottom<dim>::vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = RockBoundaryValuesBottom<dim>::value (p, c);
    }


    template <int dim>
    class RockBoundaryValuesSides : public Function<dim>
    {
    public:
      RockBoundaryValuesSides () : Function<dim>(dim+1) {}
      virtual double value (const Point<dim>   &p,
                            const unsigned int  component = 0) const;
      virtual void vector_value (const Point<dim> &p,
                                 Vector<double>   &value) const;
    };
    template <int dim>
    double
    RockBoundaryValuesSides<dim>::value (const Point<dim>  &p,
                                const unsigned int component) const
    {
			if (component == 0)
			return p[1]*std::sin(p[0]);
			else if (component == 1)
			return -p[0]*p[1]*p[1];
    		if (component == dim)
    			return p[0]*p[1];
      return 0.0;
    }

    template <int dim>
    void
    RockBoundaryValuesSides<dim>::vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = RockBoundaryValuesSides<dim>::value (p, c);
    }




    template <int dim>
    class RockRightHandSide : public Function<dim>
    {
    public:
      RockRightHandSide () : Function<dim>(dim+1) {}
      virtual double value (const Point<dim>   &p,
                            const unsigned int  component = 0) const;
      virtual void vector_value (const Point<dim> &p,
                                 Vector<double>   &value) const;
    };

    template <int dim>
    double
    RockRightHandSide<dim>::value (const Point<dim>  &p,
                               const unsigned int component) const
    {

		if (component == 0)
			return (1.0-data::phi)*(-2*p[1]*std::sin(p[0]) - 3*p[1]);
		else if (component == 1)
			return (1.0-data::phi)*(std::cos(p[0]) - 5*p[0]);
		else if (component == dim)
			return (1.0-data::phi)*(p[1]*std::cos(p[0]) - 2*p[0]*p[1]);
  return 0.0;
    }


    template <int dim>
    void
    RockRightHandSide<dim>::vector_value (const Point<dim> &p,
                                      Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = RockRightHandSide<dim>::value (p, c);
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
      values(0) = p[1]*std::sin(p[0]);
      values(1) = -p[0]*p[1]*p[1];
      values(2) = p[0]*p[1];
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
    double PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                               const unsigned int /*component*/) const
    {
        // This is the dirichlet condition for the top
        return -2./3*data::rho_f + p[0]*0.0; // last term added to avoid warnings on work comp
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
    void
    ExactSolution_pf<dim>::vector_value (const Point<dim> &p,
                                         Vector<double>   &values) const
    {
        Assert (values.size() == dim+1,
                ExcDimensionMismatch (values.size(), dim+1));


        const double permeability = data::perm_const;

        values(0) = 0.0;
        values(1) = data::lambda*data::rho_f*(1-p[1]*p[1])*permeability;
        values(2) = -data::rho_f*(p[1] - (1.0/3.0)*p[1]*p[1]*p[1]);
    }

    template <int dim>
    void
    ExactSolution_vf<dim>::vector_value (const Point<dim> &p,
                                         Vector<double>   &values) const
    {
        values(0) = 0.5*(p[0]*p[0]+p[1]*p[1]);
        values(1) = /*(1.0-1.0/data::phi)**/data::lambda*data::perm_const*data::rho_f*p[1]*p[1] - p[0]*p[1];

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

            const double permeability = data::perm_const;

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
                const double permeability = data::perm_const;

                for (unsigned int d=0; d<dim; ++d)
                    values[p][d][d] = permeability;
            }

        }
    }


    template <int dim>
    class PhiBoundaryValues : public Function<dim>
    {
    public:
        PhiBoundaryValues () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<double>            &values,
                                 const unsigned int              component = 0) const;
    };
    template <int dim>
    double
    PhiBoundaryValues<dim>::value (const Point<dim>   &p,
                                const unsigned int  component) const
    {
        const double time = this->get_time();
        return p[0]*std::exp(-p[1]) + time;
    }


    template <int dim>
    void
    PhiBoundaryValues<dim>::value_list (const std::vector<Point<dim> > &points,
                                     std::vector<double>            &values,
                                     const unsigned int              component) const
    {
        Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
        for (unsigned int i=0; i<points.size(); ++i)
            values[i] = PhiBoundaryValues<dim>::value (points[i], component);
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
    class vfInitialFunction : public Function<dim>
    {
    public:
        vfInitialFunction () : Function<dim>(dim) {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };


    template <int dim>
    double
    vfInitialFunction<dim>::value (const Point<dim>  &p,
                                   const unsigned int component) const
    {

        if (component == 0)
            return 0.0;
        else if (component == 1)
            return 0.0*p[0];
        return 0.0;
    }


    template <int dim>
    void
    vfInitialFunction<dim>::vector_value (const Point<dim> &p,
                                          Vector<double>   &values) const
    {
        for (unsigned int c=0; c<this->n_components; ++c)
            values(c) = vfInitialFunction<dim>::value (p, c);
    }

    template <int dim>
    class pfInitialFunction : public Function<dim>
    {
    public:
        pfInitialFunction () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };

    template <int dim>
    double pfInitialFunction<dim>::value (const Point<dim>  &p,
                                           const unsigned int /*component*/) const
    {

        return 0.0;
    }



    template <int dim>
    DAE<dim>::DAE (const unsigned int degree)
    :
    pr_degree (degree),
    rock_fe (FE_Q<dim>(pr_degree+1), dim,
    		FE_Q<dim>(pr_degree), 1),
    rock_dof_handler (triangulation),

	pf_degree (degree),
    pf_fe (FE_RaviartThomas<dim>(pf_degree), 1,
           FE_Q<dim>(pf_degree), 1),
    pf_dof_handler (triangulation),

    vf_degree (degree),
    vf_fe (FE_Q<dim>(pf_degree+1), dim),
    vf_dof_handler (triangulation),

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


        triangulation.refine_global (data::refinement_level);

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

        {
            rock_constraints.clear ();

            FEValuesExtractors::Vector velocities(0);
            FEValuesExtractors::Scalar pressure (dim);

            DoFTools::make_hanging_node_constraints (rock_dof_handler,
                                                     rock_constraints);

//            VectorTools::interpolate_boundary_values (rock_dof_handler,
//                                                      1,
//                                                      RockBoundaryValuesTop<dim>(),
//                                                      rock_constraints,
//                                                      rock_fe.component_mask(velocities));

//            VectorTools::interpolate_boundary_values (rock_dof_handler,
//                                                      2,
//                                                      RockBoundaryValuesBottom<dim>(),
//                                                      rock_constraints,
//                                                      rock_fe.component_mask(velocities));

//            VectorTools::interpolate_boundary_values (rock_dof_handler,
//                                                      0,
//                                                      RockBoundaryValuesSides<dim>(),
//                                                      rock_constraints,
//                                                      rock_fe.component_mask(velocities));

            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (0);
            VectorTools::compute_no_normal_flux_constraints (rock_dof_handler, 0,
                                                             no_normal_flux_boundaries,
                                                             rock_constraints);
//            std::set<types::boundary_id> no_tangential_flux_boundaries;
//            no_tangential_flux_boundaries.insert (2);
//            VectorTools::compute_nonzero_tangential_flux_constraints (rock_dof_handler, 0,
//                                                             no_tangential_flux_boundaries,
//															 ZeroFunction<dim>(dim),
//                                                             rock_constraints);

//


        }


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

        {
            BlockDynamicSparsityPattern rock_dsp (2,2);

            rock_dsp.block(0,0).reinit (n_vr, n_vr);
            rock_dsp.block(1,0).reinit (n_pr, n_vr);
            rock_dsp.block(0,1).reinit (n_vr, n_pr);
            rock_dsp.block(1,1).reinit (n_pr, n_pr);

            rock_dsp.collect_sizes();

            DoFTools::make_sparsity_pattern (rock_dof_handler, rock_dsp, rock_constraints, false);
            rock_sparsity_pattern.copy_from (rock_dsp);
        }



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
    void DAE<dim>::setup_fluid_dofs ()
	{
        pf_dof_handler.distribute_dofs (pf_fe);
            vf_dof_handler.distribute_dofs (vf_fe);


            // fluid motion numbering
            DoFRenumbering::component_wise (pf_dof_handler);
            std::vector<types::global_dof_index> dofs_per_component (dim+1);
            DoFTools::count_dofs_per_component (pf_dof_handler, dofs_per_component);
            const unsigned int n_u = dofs_per_component[0],
            					n_pf = dofs_per_component[dim],
    							n_vf = vf_dof_handler.n_dofs();


            std::cout
            << "	Number of degrees of freedom in fluid problem: "
            << pf_dof_handler.n_dofs()
            << " (" << n_u << '+' << n_pf << '+' << n_vf << ')'
            << std::endl;




                    	BlockDynamicSparsityPattern pf_dsp(2, 2);
                    	pf_dsp.block(0, 0).reinit (n_u, n_u);
                    	pf_dsp.block(1, 0).reinit (n_pf, n_u);
                    	pf_dsp.block(0, 1).reinit (n_u, n_pf);
                    	pf_dsp.block(1, 1).reinit (n_pf, n_pf);
                    	pf_dsp.collect_sizes ();
                    	DoFTools::make_sparsity_pattern (pf_dof_handler, pf_dsp, pf_constraints, false);

                    	pf_sparsity_pattern.copy_from(pf_dsp);
                    	pf_system_matrix.reinit (pf_sparsity_pattern);

                    	pf_solution.reinit (2);
                    	pf_solution.block(0).reinit (n_u);
                    	pf_solution.block(1).reinit (n_pf);
                    	pf_solution.collect_sizes ();

                    	pf_initial_solution.reinit (2);
                    	pf_initial_solution.block(0).reinit (n_u);
                    	pf_initial_solution.block(1).reinit (n_pf);
                    	pf_initial_solution.collect_sizes ();

                    	pf_system_rhs.reinit (2);
                    	pf_system_rhs.block(0).reinit (n_u);
                    	pf_system_rhs.block(1).reinit (n_pf);
                    	pf_system_rhs.collect_sizes ();



                vf_system_matrix.clear ();
                vf_constraints.clear ();
                DoFTools::make_hanging_node_constraints (vf_dof_handler,
                                                         vf_constraints);

                vf_constraints.close();
                DynamicSparsityPattern vf_dsp(vf_dof_handler.n_dofs());
                DoFTools::make_sparsity_pattern(vf_dof_handler,
                                                vf_dsp,
                                                vf_constraints,
                                                /*keep_constrained_dofs = */ true);
                vf_sparsity_pattern.copy_from(vf_dsp);

                vf_mass_matrix.reinit (vf_sparsity_pattern);
                vf_system_matrix.reinit (vf_sparsity_pattern);
                vf_solution.reinit (vf_dof_handler.n_dofs());
                vf_initial_solution.reinit (vf_dof_handler.n_dofs());
                vf_system_rhs.reinit (vf_dof_handler.n_dofs());


                MatrixCreator::create_mass_matrix(vf_dof_handler,
                                                  QGauss<dim>(vf_fe.degree+2),
                                                  vf_mass_matrix);

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
                                            dsp,
                                            phi_hanging_node_constraints,
                                            /*keep_constrained_dofs = */ true);
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

    	        FEValues<dim> pf_fe_values (pf_fe, quadrature_formula,
    	                                 update_values    | update_quadrature_points  |
    	                                 update_JxW_values | update_gradients);

    	        FEValues<dim> vf_fe_values (vf_fe, quadrature_formula,
    	                                 update_values    | update_quadrature_points  |
    	                                 update_JxW_values | update_gradients);

    	        const unsigned int   dofs_per_cell   = rock_fe.dofs_per_cell;
    	        const unsigned int   n_q_points      = quadrature_formula.size();
    	        const unsigned int   n_face_q_points = face_quadrature_formula.size();

    	        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    	        Vector<double>       local_rhs (dofs_per_cell);

    	        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    	        std::vector<double> boundary_values (n_face_q_points);
    	        const RockRightHandSide<dim>          right_hand_side;
    	    	std::vector<Vector<double> >     	  rhs_values (n_q_points, Vector<double>(dim+1));

    		    std::vector<Tensor<1,dim>>    unitz_values (n_q_points);
    		    std::vector<Tensor<1,dim>> 	  vf_values (n_q_points);
    		    std::vector<double>			  pf_values (n_q_points);
    		    std::vector<double>			  phi_values (n_q_points);
    		    std::vector<double>			  div_vf_values (n_q_points);
    		    std::vector<Tensor<1,dim>>    grad_pf_values (n_q_points);
    		    std::vector<Tensor<1,dim>>    grad_phi_values (n_q_points);

    	        const FEValuesExtractors::Vector velocities (0);
    	        const FEValuesExtractors::Scalar pressure (dim);

    	        std::vector<Tensor<1,dim>>           phi_u       (dofs_per_cell); // why is this a tensor?
    	        std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);

    	        std::vector<Tensor<2,dim> >          grad_phi_u (dofs_per_cell);
    	        std::vector<double>                  div_phi_u   (dofs_per_cell);
    	        std::vector<double>                  phi_p       (dofs_per_cell);

    	        typename DoFHandler<dim>::active_cell_iterator
    	        cell = rock_dof_handler.begin_active(),
    	        endc = rock_dof_handler.end();
    	        typename DoFHandler<dim>::active_cell_iterator
    	        phi_cell = phi_dof_handler.begin_active();
    	        typename DoFHandler<dim>::active_cell_iterator
    	        pf_cell = pf_dof_handler.begin_active();
    	        typename DoFHandler<dim>::active_cell_iterator
    	        vf_cell = vf_dof_handler.begin_active();
    	        for (; cell!=endc; ++cell, ++phi_cell, ++pf_cell, ++vf_cell)
    	        {
    	            rock_fe_values.reinit (cell);
    	            phi_fe_values.reinit (phi_cell);
    	            pf_fe_values.reinit (pf_cell);
//    	            vf_fe_values.reinit (vf_cell);

    	            local_matrix = 0;
    	            local_rhs = 0;

    	            right_hand_side.vector_value_list(rock_fe_values.get_quadrature_points(),
    	                                                 rhs_values);

    	            phi_fe_values.get_function_values (phi_solution, phi_values);
    	            phi_fe_values.get_function_gradients (phi_solution, grad_phi_values);
    	            pf_fe_values[pressure].get_function_values (pf_solution, pf_values);
    	            pf_fe_values[pressure].get_function_gradients (pf_solution, grad_pf_values);



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
    	                                rhs_values[q](component_i) *
    	                                rock_fe_values.JxW(q);
    	                    }
    	                    else
    	                    {

    	                    	local_rhs(i) += (grad_phi_values[q]*phi_u[i]
													 + ((1.0-phi_values[1])*data::rho_r + phi_values[q]*data::rho_f)
													 * unitz_values[q]*phi_u[i]
																		  - phi_values[q]*phi_p[i]
																							)*
														rock_fe_values.JxW(q);
    	                    	std::cout << phi_values[q] << std::endl;
//
    	                    }

    	                }

    	              }


    	              cell->get_dof_indices (local_dof_indices);
    	              rock_constraints.distribute_local_to_global (local_matrix, local_rhs,
    	                                                      local_dof_indices,
    	                                                      rock_system_matrix, rock_system_rhs);


    	        }

//    	        std::map<types::global_dof_index, double> boundary_values_flux;
//    	        {
//    	                types::global_dof_index n_dofs = rock_dof_handler.n_dofs();
//    	                std::vector<bool> componentVector(dim + 1, false); // condition is on pressue
//    	                // setting flux value for the sides at 0 ON THE PRESSURE
//    	                componentVector[dim] = true;
//
//    	                std::vector<bool> selected_dofs(n_dofs);
//    	                std::set< types::boundary_id > boundary_ids;
//    	                boundary_ids.insert(2);
//
//    	                DoFTools::extract_boundary_dofs(rock_dof_handler, ComponentMask(componentVector),
//    	                        selected_dofs, boundary_ids);
//
//    	                for (types::global_dof_index i = 0; i < n_dofs; i++) {
//    	                    if (selected_dofs[i]) boundary_values_flux[i] = 00.0;
//    	                }
//
//    	        }


//    	        MatrixTools::apply_boundary_values(boundary_values_flux,
//    	                rock_system_matrix, rock_solution, rock_system_rhs);




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
    void DAE<dim>::assemble_pf_system ()
    {
        QGauss<dim>   quadrature_formula(pf_degree+2);
        QGauss<dim-1> face_quadrature_formula(pf_degree+2);

        FEValues<dim> pf_fe_values (pf_fe, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> pf_fe_face_values (pf_fe, face_quadrature_formula,
                                             update_values    | update_normal_vectors |
                                             update_quadrature_points  | update_JxW_values);

        FEValues<dim> vr_fe_values (rock_fe, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> vr_fe_face_values (rock_fe, face_quadrature_formula,
                                                  update_values | update_quadrature_points |
                                                  update_JxW_values | update_normal_vectors);

        const unsigned int   dofs_per_cell   = pf_fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();

        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

        const PressureBoundaryValues<dim> pressure_boundary_values;
        const KInverse<dim>               k_inverse;
        const K<dim>               		  k;

        std::vector<double> div_vr_values (n_q_points);
        std::vector<double> boundary_values (n_face_q_points);
	    std::vector<double> pr_face_values (n_face_q_points);

        std::vector<Tensor<2,dim> > k_inverse_values (n_q_points);
        std::vector<Tensor<2,dim> > k_values (n_q_points);


        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);

        typename DoFHandler<dim>::active_cell_iterator
        cell = pf_dof_handler.begin_active(),
        endc = pf_dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = rock_dof_handler.begin_active();
        for (; cell!=endc; ++cell, ++vr_cell)
        {
            pf_fe_values.reinit (cell);
            vr_fe_values.reinit(vr_cell);

            local_matrix = 0;
            local_rhs = 0;

            k_inverse.value_list (pf_fe_values.get_quadrature_points(),
                                  k_inverse_values);
            k.value_list (pf_fe_values.get_quadrature_points(),
                          k_values);

            vr_fe_values[velocities].get_function_divergences (rock_solution, div_vr_values);

            for (unsigned int q=0; q<n_q_points; ++q)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    const Tensor<1,dim> phi_i_u     = pf_fe_values[velocities].value (i, q);
                    const double        div_phi_i_u = pf_fe_values[velocities].divergence (i, q);
                    const double        phi_i_p     = pf_fe_values[pressure].value (i, q);

                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                        const Tensor<1,dim> phi_j_u     = pf_fe_values[velocities].value (j, q);
                        const double        div_phi_j_u = pf_fe_values[velocities].divergence (j, q);
                        const double        phi_j_p     = pf_fe_values[pressure].value (j, q);

                        local_matrix(i,j) += ( 1./data::lambda * phi_i_u * k_inverse_values[q] * phi_j_u
                                              - div_phi_i_u * phi_j_p
                                              - phi_i_p * div_phi_j_u)
                        * pf_fe_values.JxW(q);
                    }

                    local_rhs(i) += phi_i_p *
                    					div_vr_values[q] *
										pf_fe_values.JxW(q);
                    // BE CAREFUL HERE ONCE K is not constant or 1 anymore. have taken div of the constants as zero
                }


            for (unsigned int face_no=0;
                 face_no<GeometryInfo<dim>::faces_per_cell;
                 ++face_no)
                if (cell->face(face_no)->at_boundary()
                    &&
                    (cell->face(face_no)->boundary_id() == 1)) // Basin top has boundary id 1
                {
                    pf_fe_face_values.reinit (cell, face_no);
                    vr_fe_face_values.reinit (vr_cell, face_no);
                    vr_fe_face_values[pressure].get_function_values (rock_solution, pr_face_values);

                    // DIRICHLET CONDITION FOR TOP. pf = pr at top of basin

                    pressure_boundary_values
                    .value_list (pf_fe_face_values.get_quadrature_points(),
                                 boundary_values);

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                            local_rhs(i) += -( pf_fe_face_values[velocities].value (i, q) *
                                              pf_fe_face_values.normal_vector(q) *
                                              pr_face_values[q] *
                                              pf_fe_face_values.JxW(q));
                }


            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    pf_system_matrix.add (local_dof_indices[i],
                                          local_dof_indices[j],
                                          local_matrix(i,j));
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                pf_system_rhs(local_dof_indices[i]) += local_rhs(i);
        }


        // NOW NEED TO STRONGLY IMPOSE THE FLUX CONDITIONS BOTH ON SIDES AND BOTTOM

        std::map<types::global_dof_index, double> boundary_values_flux;
        {
            types::global_dof_index n_dofs = pf_dof_handler.n_dofs();
            std::vector<bool> componentVector(dim + 1, true); // condition is on pressue
            // setting flux value for the sides at 0 ON THE PRESSURE
            componentVector[dim] = false;
            std::vector<bool> selected_dofs(n_dofs);
            std::set< types::boundary_id > boundary_ids;
            boundary_ids.insert(0);

            DoFTools::extract_boundary_dofs(pf_dof_handler, ComponentMask(componentVector),
                                            selected_dofs, boundary_ids);

            for (types::global_dof_index i = 0; i < n_dofs; i++) {
                if (selected_dofs[i]) boundary_values_flux[i] = 0.0; // Side boudaries have flux 0 on pressure
            }

        }

        // Apply the conditions
        MatrixTools::apply_boundary_values(boundary_values_flux,
                                           pf_system_matrix, pf_solution, pf_system_rhs);


    }

    template <int dim>
    void DAE<dim>::assemble_vf_system ()
    {

        QGauss<dim>   quadrature_formula(vf_degree+2);

        FEValues<dim> vf_fe_values (vf_fe, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        FEValues<dim> pf_fe_values (pf_fe, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);

        FEValues<dim> vr_fe_values (rock_fe, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);

        FEValues<dim> phi_fe_values (phi_fe, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);


        const unsigned int   dofs_per_cell   = vf_fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();

        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	       std::vector<Tensor<1,dim>>     grad_pf_values (n_q_points);
	       std::vector<Tensor<1,dim>>     unitz_values (n_q_points);
	       std::vector<Tensor<1,dim>> 	  vr_values (n_q_points);
	       std::vector<double>			  phi_values (n_q_points);

	       const FEValuesExtractors::Vector velocities (0);
	       const FEValuesExtractors::Scalar pressure (dim);

        typename DoFHandler<dim>::active_cell_iterator
        cell = vf_dof_handler.begin_active(),
        endc = vf_dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator
        pf_cell = pf_dof_handler.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = rock_dof_handler.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        phi_cell = phi_dof_handler.begin_active();
        for (; cell!=endc; ++cell, ++pf_cell, ++vr_cell, ++phi_cell)
	     	 {
                 vf_fe_values.reinit(cell);
                 pf_fe_values.reinit(pf_cell);
                 vr_fe_values.reinit(vr_cell);
                 phi_fe_values.reinit(phi_cell);

                 local_matrix = 0;
                 local_rhs = 0;

                 unitz (vf_fe_values.get_quadrature_points(), unitz_values);
                 pf_fe_values[pressure].get_function_gradients (pf_solution, grad_pf_values);
                 vr_fe_values[velocities].get_function_values (rock_solution, vr_values);
                 phi_fe_values.get_function_values (phi_solution, phi_values);

                 for (unsigned int i=0; i<dofs_per_cell; ++i)
                 {
                     const unsigned int
                     component_i = vf_fe.system_to_component_index(i).first;

                     for (unsigned int q=0; q<n_q_points; ++q)
                         local_rhs(i) +=    (-vf_fe_values.shape_value(i,q) *
                                             data::lambda*data::perm_const*(1.0/phi_values[q])* data::rho_f*
                                             unitz_values[q][component_i] +
                                             vf_fe_values.shape_value(i,q) *
                                             vr_values[q][component_i]  // rock velocity added
                                             - vf_fe_values.shape_value(i,q) *
                                             data::lambda*data::perm_const*(1.0/phi_values[q])* grad_pf_values[q][component_i] // minus pressure gradient
                                             ) *
											 	 	 vf_fe_values.JxW(q);

                 }

                 cell->get_dof_indices (local_dof_indices);
                 for (unsigned int i=0; i<dofs_per_cell; ++i)
                     for (unsigned int j=0; j<dofs_per_cell; ++j)
                         vf_system_matrix.add (local_dof_indices[i],
                                               local_dof_indices[j],
                                               local_matrix(i,j));

                 for (unsigned int i=0; i<dofs_per_cell; ++i)
                     vf_system_rhs(local_dof_indices[i]) += local_rhs(i);
             }
        vf_constraints.condense (vf_system_matrix);
        vf_constraints.condense (vf_system_rhs);

//        std::map<types::global_dof_index, double> vf_boundary;
//        {
//            types::global_dof_index n_dofs = vf_dof_handler.n_dofs();
//            std::vector<bool> componentVector(dim, true);
//                                  componentVector[0] = false;
//            std::vector<bool> selected_dofs(n_dofs);
//            std::set< types::boundary_id > boundary_ids;
//            boundary_ids.insert(2);
//
//            DoFTools::extract_boundary_dofs(vf_dof_handler, ComponentMask(componentVector),
//                                            selected_dofs, boundary_ids);
//
//            for (types::global_dof_index i = 0; i < n_dofs; i++) {
//                if (selected_dofs[i]) vf_boundary[i] = 00.0;
//            }
//
//        }


        vf_system_matrix.copy_from(vf_mass_matrix);
//        MatrixTools::apply_boundary_values(vf_boundary,
//                                           vf_system_matrix, vf_solution, vf_system_rhs);
    }


    template <int dim>
    void DAE<dim>::solve_fluid_system ()
    {
//        std::cout << "   Solving for p_f..." << std::endl;
        SparseDirectUMFPACK  pf_direct;
        pf_direct.initialize(pf_system_matrix);
        pf_direct.vmult (pf_solution, pf_system_rhs);

        assemble_vf_system ();
//        std::cout << "   Solving for v_f..." << std::endl;
        SparseDirectUMFPACK  vf_direct;
        vf_direct.initialize(vf_system_matrix);
        vf_direct.vmult (vf_solution, vf_system_rhs);



    }



    template <int dim>
    void
    DAE<dim>::assemble_phi_system ()
    {
        phi_system_matrix = 0;
        phi_nontime_matrix = 0;

        MatrixCreator::create_mass_matrix(phi_dof_handler,
                                          QGauss<dim>(data::problem_degree+4),
                                          phi_mass_matrix);

        FullMatrix<double>                   cell_matrix;

        std::vector<types::global_dof_index> local_dof_indices;

        QGauss<dim>  quadrature_formula(data::problem_degree+3);
        QGauss<dim-1> face_quadrature_formula(data::problem_degree+3);


        FEValues<dim> phi_fe_values (phi_fe, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> phi_fe_face_values (phi_fe, face_quadrature_formula,
                                          update_values | update_quadrature_points |
                                          update_JxW_values | update_normal_vectors);
        FEValues<dim> vr_fe_values (rock_fe, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> vr_fe_face_values (rock_fe, face_quadrature_formula,
                                                  update_values | update_quadrature_points |
                                                  update_JxW_values | update_normal_vectors);

        const unsigned int dofs_per_cell   = phi_fe.dofs_per_cell;
        const unsigned int n_q_points      = phi_fe_values.get_quadrature().size();
        const unsigned int n_face_q_points = phi_fe_face_values.get_quadrature().size();


        cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);

	       std::vector<Tensor<1,dim>> 	  vr_values (n_q_points);
	       std::vector<Tensor<1,dim>> 	  vr_face_values (n_face_q_points);

	       const FEValuesExtractors::Vector velocities (0);
	       const FEValuesExtractors::Scalar pressure (dim);


        typename DoFHandler<dim>::active_cell_iterator
        cell = phi_dof_handler.begin_active(),
        endc = phi_dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = rock_dof_handler.begin_active();
        for (; cell!=endc; ++cell, ++vr_cell)
        {
            cell_matrix = 0;

            phi_fe_values.reinit (cell);
            vr_fe_values.reinit (vr_cell);
            vr_fe_values[velocities].get_function_values (rock_solution, vr_values);

            const double delta = 0.1 * cell->diameter ();
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        cell_matrix(i,j) += ((vr_values[q_point] *
                                              phi_fe_values.shape_grad(j,q_point)   *
                                              (phi_fe_values.shape_value(i,q_point) +
                                               0.0*delta *
                                               (vr_values[q_point] *
                                                phi_fe_values.shape_grad(i,q_point)))) *
                                             phi_fe_values.JxW(q_point));
                }

            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary())
                {
                    phi_fe_face_values.reinit (cell, face);
                    vr_fe_face_values.reinit (vr_cell, face);
                    vr_fe_face_values[velocities].get_function_values (rock_solution, vr_face_values);

                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                        if (phi_fe_face_values.normal_vector(q_point) *
                            vr_face_values[q_point]
                            < 0)
                            for (unsigned int i=0; i<dofs_per_cell; ++i)
                            {
                                for (unsigned int j=0; j<dofs_per_cell; ++j)
                                    cell_matrix(i,j) -= (vr_face_values[q_point] *
                                                         phi_fe_face_values.normal_vector(q_point) *
                                                         phi_fe_face_values.shape_value(i,q_point) *
                                                         phi_fe_face_values.shape_value(j,q_point) *
                                                         phi_fe_face_values.JxW(q_point));

                            }
                }

            cell->get_dof_indices (local_dof_indices);

            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                    phi_nontime_matrix.add (local_dof_indices[i],
                                        local_dof_indices[j],
                                        cell_matrix(i,j));
            }
        }

    }


    template <int dim>
    void
    DAE<dim>::assemble_phi_rhs ()
    {
        phi_system_rhs=0;
        phi_nontime_rhs=0;

        Vector<double>                       cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;

        QGauss<dim>  quadrature_formula(data::problem_degree+4);
        QGauss<dim-1> face_quadrature_formula(data::problem_degree+4);


        FEValues<dim> phi_fe_values (phi_fe, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> phi_fe_face_values (phi_fe, face_quadrature_formula,
                                          update_values | update_quadrature_points |
                                          update_JxW_values | update_normal_vectors);
        FEValues<dim> vr_fe_values (rock_fe, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> vr_fe_face_values (rock_fe, face_quadrature_formula,
                                                  update_values | update_quadrature_points |
                                                  update_JxW_values | update_normal_vectors);
        FEValues<dim> pf_fe_values (pf_fe, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);

        PhiBoundaryValues<dim> phi_boundary_values;
        phi_boundary_values.set_time(time);

        const unsigned int dofs_per_cell   = phi_fe.dofs_per_cell;
        const unsigned int n_q_points      = phi_fe_values.get_quadrature().size();
        const unsigned int n_face_q_points = phi_fe_face_values.get_quadrature().size();


        cell_rhs.reinit (dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);

        std::vector<double>         rhs_values (n_q_points);
        std::vector<double>         face_phi_boundary_values (n_face_q_points);

	       const FEValuesExtractors::Vector velocities (0);
	       const FEValuesExtractors::Scalar pressure (dim);
	       std::vector<Tensor<1,dim>> 	  vr_values (n_q_points);
	       std::vector<Tensor<1,dim>> 	  vr_face_values (n_face_q_points);
	       std::vector<double>			  pr_values (n_q_points);
	       std::vector<double>			  pf_values (n_q_points);

        typename DoFHandler<dim>::active_cell_iterator
        cell = phi_dof_handler.begin_active(),
        endc = phi_dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = rock_dof_handler.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        pf_cell = pf_dof_handler.begin_active();
        for (; cell!=endc; ++cell, ++vr_cell, ++pf_cell)
        {
            cell_rhs = 0;
            phi_fe_values.reinit (cell);
            vr_fe_values.reinit (vr_cell);
            pf_fe_values.reinit (pf_cell);
            vr_fe_values.reinit (vr_cell);

            vr_fe_values[velocities].get_function_values (rock_solution, vr_values);
            vr_fe_values[pressure].get_function_values (rock_solution, pr_values);
            pf_fe_values[pressure].get_function_values (pf_solution, pf_values);

            const double delta = 0.1 * cell->diameter ();
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                	const double rhs_value = -data::gamma * std::exp(-data::gamma*(pr_values[q_point]-pf_values[q_point]));

                    cell_rhs(i) += ((phi_fe_values.shape_value(i,q_point) +
                                     delta *
                                     (vr_values[q_point] *
                                      phi_fe_values.shape_grad(i,q_point))        ) *
                                    rhs_value *
                                    phi_fe_values.JxW (q_point));
                }

            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary())
                {
                    phi_fe_face_values.reinit (cell, face);
                    vr_fe_face_values.reinit (vr_cell, face);
                    phi_boundary_values.value_list (phi_fe_face_values.get_quadrature_points(),
                                                face_phi_boundary_values);
                    vr_fe_face_values[velocities].get_function_values (rock_solution, vr_face_values);

                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                        if (phi_fe_face_values.normal_vector(q_point) *
                            vr_face_values[q_point]
                            < 0)
                            for (unsigned int i=0; i<dofs_per_cell; ++i)
                            {
                                cell_rhs(i) -= (vr_face_values[q_point] *
                                                phi_fe_face_values.normal_vector(q_point) *
                                                face_phi_boundary_values[q_point]         *
                                                phi_fe_face_values.shape_value(i,q_point) *
                                                phi_fe_face_values.JxW(q_point));

                            }
                }

            cell->get_dof_indices (local_dof_indices);


            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                phi_nontime_rhs(local_dof_indices[i]) += cell_rhs(i);
            }
        }
    }


    template <int dim>
    void DAE<dim>::solve_phi ()
    {
        SparseDirectUMFPACK  A_direct;
        A_direct.initialize(phi_system_matrix);
        A_direct.vmult (phi_solution, phi_system_rhs);


        phi_hanging_node_constraints.distribute (phi_solution);
    }


    template <int dim>
    void DAE<dim>::compute_errors () const
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
  	  std::cout << "Errors: ||e_pr||_L2 = " << p_l2_error
  	            << ",   ||e_vr||_L2 = " << u_l2_error
  	            << std::endl;
//
//        const ComponentSelectFunction<dim>
//        pressure_mask (dim, dim+1);
//        const ComponentSelectFunction<dim>
//        velocity_mask(std::make_pair(0, dim), dim+1);
//
//        ExactSolution_pf<dim> exact_pf_solution;
//        Vector<double> cellwise_errors (triangulation.n_active_cells());
//
//        QTrapez<1>     q_trapez;
//        QIterated<dim> quadrature (q_trapez, pf_degree+2);
//
//        VectorTools::integrate_difference (pf_dof_handler, pf_solution, exact_pf_solution,
//                                           cellwise_errors, quadrature,
//                                           VectorTools::L2_norm,
//                                           &pressure_mask);
//        const double pf_l2_error = cellwise_errors.l2_norm();
//
//        VectorTools::integrate_difference (pf_dof_handler, pf_solution, exact_pf_solution,
//                                           cellwise_errors, quadrature,
//                                           VectorTools::L2_norm,
//                                           &velocity_mask);
//        const double u_l2_error = cellwise_errors.l2_norm();
//
//
//        ExactSolution_vf<dim> exact_solution_vf;
//        Vector<double> cellwise_errors_vf (triangulation.n_active_cells());
//
//
//        VectorTools::integrate_difference (vf_dof_handler, vf_solution, exact_solution_vf,
//                                           cellwise_errors_vf, quadrature,
//                                           VectorTools::L2_norm);
//        const double vf_l2_error = cellwise_errors_vf.l2_norm();
//
//        std::cout << "Errors: ||e_pf||_L2 = " << pf_l2_error
//        << "," << std::endl <<   "        ||e_u||_L2 = " << u_l2_error
//        << ", " << std::endl <<   "        ||e_vf||_L2 = " << vf_l2_error
//        << ". " << std::endl;

    }



    template <int dim>
    void DAE<dim>::output_results () const
    {
    	{
            std::vector<std::string> vr_names (dim, "v_r");
            vr_names.push_back ("p_r");

            std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation
            (dim, DataComponentInterpretation::component_is_part_of_vector);
            data_component_interpretation
            .push_back (DataComponentInterpretation::component_is_scalar);

            DataOut<dim> data_out;
            data_out.attach_dof_handler (rock_dof_handler);
            data_out.add_data_vector (rock_solution, vr_names,
                                      DataOut<dim>::type_dof_data,
                                      data_component_interpretation);
            data_out.build_patches ();

            const std::string filename = "rock_solution-"
            + Utilities::int_to_string(timestep_number, 3) +
            ".vtk";

            std::ofstream output (filename.c_str());
            data_out.write_vtk (output);

    	}
    	{
        std::vector<std::string> pf_names (dim, "uf");
        pf_names.push_back ("p_f");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        pf_component_interpretation
        (dim+1, DataComponentInterpretation::component_is_scalar);
        for (unsigned int i=0; i<dim; ++i)
            pf_component_interpretation[i]
            = DataComponentInterpretation::component_is_part_of_vector;
        DataOut<dim> data_out;

        data_out.add_data_vector (pf_dof_handler, pf_solution,
                                  pf_names, pf_component_interpretation);
        data_out.add_data_vector (vf_dof_handler, vf_solution,
                                  "v_f");

        data_out.build_patches ();
            const std::string filename = "fluid_solution-"
            + Utilities::int_to_string(timestep_number, 3) +
            ".vtk";
        std::ofstream output (filename.c_str());
        data_out.write_vtk (output);
    	}

    }

    template <int dim>
    void DAE<dim>::output_phi_results ()
    {

        DataOut<dim> data_out;

        data_out.attach_dof_handler(phi_dof_handler);
        data_out.add_data_vector(phi_solution, "U");

        data_out.build_patches();

        const std::string filename = "phi_solution-"
        + Utilities::int_to_string(timestep_number, 3) +
        ".vtk";
        std::ofstream output(filename.c_str());
        data_out.write_vtk(output);


    }


    template <int dim>
    void DAE<dim>::run ()
    {
    	timestep_number = 0;
    	time            = 0;

        make_grid ();
        setup_rock_dofs ();
        setup_fluid_dofs ();
        setup_phi_dofs ();

	       const FEValuesExtractors::Vector velocities (0);
	       const FEValuesExtractors::Scalar pressure (dim);

        VectorTools::interpolate(phi_dof_handler, PhiInitialFunction<dim>(), old_phi_solution);
        VectorTools::interpolate(vf_dof_handler, vfInitialFunction<dim>(), vf_initial_solution);

        phi_solution = old_phi_solution;

        assemble_rock_system (); // need old_phi_solution here
        solve_rock_system ();
        assemble_pf_system ();
        solve_fluid_system ();


        assemble_phi_system ();

        output_results ();
        output_phi_results();

        time_step = data::timestep;

        while (time <= data::final_time)
                {
                    time += time_step;
                    ++timestep_number;
                    std::cout << "   Solving at timstep number "<< timestep_number <<"..." << std::endl;


                    assemble_phi_rhs ();

                    phi_mass_matrix.vmult(phi_system_rhs, old_phi_solution);
                    phi_system_rhs.add(time_step,phi_nontime_rhs);

                    phi_system_matrix.copy_from(phi_mass_matrix);
                    phi_system_matrix.add(time_step,phi_nontime_matrix);

                    phi_hanging_node_constraints.condense (phi_system_rhs);
                    phi_hanging_node_constraints.condense (phi_system_matrix);


                    solve_phi ();
                    output_phi_results ();
                    // error_analysis();
                    old_phi_solution = phi_solution;

                    assemble_rock_system (); // need old_phi_solution here
                    solve_rock_system ();
                    assemble_pf_system ();
                    solve_fluid_system ();

                    assemble_phi_system ();
                    output_results ();


                }

        compute_errors ();

    }
}


int main ()
{
    try
    {
        using namespace dealii;
        using namespace System;

        DAE<data::dimension> dae_problem(data::problem_degree);
        dae_problem.run ();
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
